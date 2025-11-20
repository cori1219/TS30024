#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sensor Logger HTTP Push → 받은 HTTP 데이터를 그대로 출력하는 서버

- POST /data 로 들어온 요청의:
  - RAW BODY (순수 텍스트)
  - JSON 파싱 결과 (가능하면)

를 터미널에 그대로 찍어준다.
"""

from flask import Flask, request, jsonify
import json

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "HTTP echo server is running.\nTry POST /data", 200


@app.route("/data", methods=["POST"])
def handle_data():
    # 1) raw body 그대로
    raw_body = request.get_data(as_text=True)

    print("\n================= NEW /data REQUEST =================")
    print(">>> RAW BODY")
    print(raw_body)

    # 2) JSON 파싱 시도
    parsed = None
    print(">>> PARSED JSON")
    try:
        parsed = request.get_json(force=True)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[WARN] JSON parse failed: {e}")
        parsed = None

    print("=====================================================\n")

    # 클라이언트 쪽에서도 확인할 수 있게 그대로 돌려줌
    return jsonify({
        "status": "ok",
        "raw_body": raw_body,
        "parsed_json": parsed,
    }), 200


if __name__ == "__main__":
    # Sensor Logger에서:
    #   URL: http://<서버IP>:8000/data
    #   Method: POST
    app.run(host="0.0.0.0", port=8000, debug=True)

