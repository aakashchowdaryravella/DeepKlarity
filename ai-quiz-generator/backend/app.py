# app.py
# Flask backend — robust routing for API + static frontend, CORS, debug endpoint
import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Google generative client
import google.generativeai as genai

# Optional: enable CORS for development
from flask_cors import CORS

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable (.env or OS).")

# Configure client
try:
    genai.configure(api_key=API_KEY)
except Exception:
    # some older/newer versions might prefer google.auth ADC; setting env vars helps
    pass

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)  # allow frontend requests during development

# ---------- Helpers ----------
def _get_models_list():
    try:
        models_iter = genai.list_models()
        models = list(models_iter)
        return models
    except Exception as e:
        raise RuntimeError(f"Failed to list models: {e}")

def choose_model(preferred_list=None):
    if preferred_list is None:
        preferred_list = [
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-ultra-1.0",
            "gemini-1.0",
            "text-bison@001",
            "text-bison",
        ]
    models = _get_models_list()
    if not models:
        raise RuntimeError("No models returned by the API (empty list).")
    available = {getattr(m, "name", str(m)).lower(): m for m in models}
    for cand in preferred_list:
        if cand.lower() in available:
            return getattr(available[cand.lower()], "name", str(available[cand.lower()]))
    for m in models:
        caps = getattr(m, "capabilities", None)
        if caps:
            low_caps = [c.lower() for c in caps]
            if any(k in low_caps for k in ("generate", "generations", "text", "chat", "responses")):
                return getattr(m, "name", str(m))
    return getattr(models[0], "name", str(models[0]))

def _parse_response_text(resp):
    try:
        if hasattr(resp, "text") and isinstance(getattr(resp, "text"), str):
            return resp.text
        if isinstance(resp, dict):
            if "output" in resp and isinstance(resp["output"], str):
                return resp["output"]
            if "outputs" in resp and isinstance(resp["outputs"], list) and resp["outputs"]:
                first = resp["outputs"][0]
                if isinstance(first, dict):
                    for k in ("content", "text", "output"):
                        if k in first and isinstance(first[k], str):
                            return first[k]
            if "candidates" in resp and isinstance(resp["candidates"], list) and resp["candidates"]:
                cand = resp["candidates"][0]
                if isinstance(cand, dict) and "output" in cand:
                    return cand["output"]
        if hasattr(resp, "candidates"):
            cands = getattr(resp, "candidates")
            if isinstance(cands, (list, tuple)) and cands:
                first = cands[0]
                if hasattr(first, "text"):
                    return first.text
                if isinstance(first, dict) and "output" in first:
                    return first["output"]
        if hasattr(resp, "outputs"):
            outs = getattr(resp, "outputs")
            if isinstance(outs, (list, tuple)) and outs:
                first = outs[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    for k in ("content", "text", "output"):
                        if k in first and isinstance(first[k], str):
                            return first[k]
        return str(resp)
    except Exception:
        return str(resp)

def generate_text_with_model(model_name, prompt, max_output_tokens=256, temperature=0.2):
    last_exc = None
    # 1) genai.generate_text
    try:
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model_name, prompt=prompt,
                                      max_output_tokens=max_output_tokens, temperature=temperature)
            return _parse_response_text(resp)
    except Exception as e:
        last_exc = e
    # 2) genai.generate
    try:
        if hasattr(genai, "generate"):
            resp = genai.generate(model=model_name, prompt=prompt,
                                  max_output_tokens=max_output_tokens, temperature=temperature)
            return _parse_response_text(resp)
    except Exception as e:
        last_exc = e
    # 3) genai.responses.create
    try:
        if hasattr(genai, "responses") and hasattr(genai.responses, "create"):
            try:
                resp = genai.responses.create(model=model_name, input=prompt,
                                              max_output_tokens=max_output_tokens, temperature=temperature)
            except TypeError:
                resp = genai.responses.create(model=model_name, prompt=prompt,
                                              max_output_tokens=max_output_tokens, temperature=temperature)
            return _parse_response_text(resp)
    except Exception as e:
        last_exc = e
    # 4) genai.create
    try:
        if hasattr(genai, "create"):
            resp = genai.create(model=model_name, prompt=prompt)
            return _parse_response_text(resp)
    except Exception as e:
        last_exc = e
    raise RuntimeError(f"All generation attempts failed. Last error: {last_exc or 'Unknown error'}")

# ---------- Routes ----------
@app.route("/api/list-models", methods=["GET"])
def list_models_route():
    try:
        models = _get_models_list()
        out = []
        for m in models:
            out.append({
                "name": getattr(m, "name", str(m)),
                "displayName": getattr(m, "displayName", ""),
                "capabilities": getattr(m, "capabilities", []),
            })
        return jsonify({"ok": True, "models": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/generate", methods=["POST"])
def generate_route():
    payload = request.json or {}
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "Missing prompt in JSON body."}), 400

    # debug log
    app.logger.info("API /api/generate called; prompt length=%d", len(prompt))

    try:
        model_name = choose_model()
    except Exception as e:
        return jsonify({"ok": False, "error": f"Model selection failed: {e}"}), 500

    try:
        output_text = generate_text_with_model(model_name, prompt, max_output_tokens=256, temperature=0.2)
        return jsonify({"ok": True, "model_used": model_name, "output": output_text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Debug endpoint to get detailed attempt errors
@app.route("/api/debug-generate", methods=["POST"])
def debug_generate():
    payload = request.json or {}
    prompt = payload.get("prompt", "").strip() or "Hello debug"
    try:
        model_name = choose_model()
    except Exception as e:
        return jsonify({"ok": False, "stage": "choose_model", "error": str(e)}), 500

    results = {"model_used": model_name, "attempts": []}
    def try_call(name, fn):
        try:
            out = fn()
            return {"method": name, "ok": True, "output_preview": (out[:500] if isinstance(out, str) else str(out))}
        except Exception as e:
            return {"method": name, "ok": False, "error": repr(e)}

    # Attempt 1
    results["attempts"].append(try_call("genai.generate_text", lambda: _parse_response_text(
        genai.generate_text(model=model_name, prompt=prompt, max_output_tokens=128, temperature=0.2)
    ) if hasattr(genai, "generate_text") else (_ for _ in ()).throw(Exception("attribute missing"))))

    # Attempt 2
    results["attempts"].append(try_call("genai.generate", lambda: _parse_response_text(
        genai.generate(model=model_name, prompt=prompt, max_output_tokens=128, temperature=0.2)
    ) if hasattr(genai, "generate") else (_ for _ in ()).throw(Exception("attribute missing"))))

    # Attempt 3
    def try_responses():
        if not (hasattr(genai, "responses") and hasattr(genai.responses, "create")):
            raise Exception("responses.create missing")
        try:
            r = genai.responses.create(model=model_name, input=prompt, max_output_tokens=128, temperature=0.2)
        except TypeError:
            r = genai.responses.create(model=model_name, prompt=prompt, max_output_tokens=128, temperature=0.2)
        return _parse_response_text(r)
    results["attempts"].append(try_call("genai.responses.create", try_responses))

    # Attempt 4
    results["attempts"].append(try_call("genai.create", lambda: _parse_response_text(genai.create(model=model_name, prompt=prompt)) if hasattr(genai, "create") else (_ for _ in ()).throw(Exception("attribute missing"))))

    return jsonify({"ok": True, "debug": results})

# Serve static frontend safely: do NOT let this catch /api/* requests
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>", methods=["GET"])
def serve_frontend(path):
    # If it's an API path, return 404 so Flask routes to correct handlers (or client gets clear response)
    if path.startswith("api/") or path.startswith("api\\"):
        return jsonify({"ok": False, "error": "Not Found"}), 404

    # serve static file if exists
    full_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)

    # otherwise serve index.html
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
