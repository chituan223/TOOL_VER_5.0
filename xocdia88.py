import websocket
import json
import threading
import time
from flask import Flask, jsonify

# ================= CONFIG =================
WS_URL = "wss://taixiumd5.system32-cloudfare-356783752985678522.monster/signalr/connect?transport=webSockets&connectionToken=BPl1xYA7PD3qBXdpCsyYkKcmVC7loSR44XtkvzIBTALxhSWti33KXi2uFAOkITLxb53COovCFQFLb4J1og5lgo06K2r7qFe%2FM3%2BWMkTfhlzkBeMnnpXT%2FIk%2BdD%2BZiaeW&connectionData=%5B%7B%22name%22%3A%22md5luckydiceHub%22%7D%5D&tid=6&access_token=YOUR_TOKEN"

app = Flask(__name__)

latest_data = {}
last_session_id = None

# ================= XỬ LÝ DATA =================
def parse_data(data):
    global latest_data, last_session_id

    try:
        msg = json.loads(data)

        if "M" not in msg:
            return

        for m in msg["M"]:
            if m.get("M") != "Md5sessionInfo":
                continue

            info = m["A"][0]
            session_id = info.get("SessionID")

            # ❌ bỏ nếu trùng phiên
            if session_id == last_session_id:
                return

            # ===== check result =====
            result = info.get("Result")
            if not result:
                return

            d1 = result.get("Dice1", -1)
            d2 = result.get("Dice2", -1)
            d3 = result.get("Dice3", -1)

            # ❌ CHẶN -1 -1 -1
            if d1 == -1 or d2 == -1 or d3 == -1:
                return

            total = d1 + d2 + d3
            ket = "Tài" if total >= 11 else "Xỉu"

            # ===== lấy tiền =====
            tai_money = info.get("TotalBetTai", 0)
            xiu_money = info.get("TotalBetXiu", 0)

            # ❌ CHẶN tiền rác
            if (
                tai_money in [-1, None] or
                xiu_money in [-1, None] or
                tai_money <= 0 or
                xiu_money <= 0
            ):
                return

            tai_money = int(tai_money)
            xiu_money = int(xiu_money)

            # ================= DỰ ĐOÁN =================
            if tai_money > xiu_money:
                du_doan = "Xỉu"   # tiền nhiều thường bị bẻ
            else:
                du_doan = "Tài"

            # ================= TỶ LỆ =================
            tong_tien = tai_money + xiu_money
            ty_le = round((abs(tai_money - xiu_money) / tong_tien) * 100, 2)

            # ✅ update phiên mới
            last_session_id = session_id

            latest_data = {
                "phien": session_id,
                "xuc_xac_1": d1,
                "xuc_xac_2": d2,
                "xuc_xac_3": d3,
                "tong": total,
                "ket_qua": ket,
                "md5": info.get("Md5Encript", ""),
                "tai_tong_tien": tai_money,
                "xiu_tong_tien": xiu_money,
                "du_doan": du_doan,
                "ty_le": ty_le
            }

            print("✅ DATA CHUẨN:", latest_data)

    except Exception as e:
        print("Parse error:", e)


# ================= WS =================
def on_message(ws, message):
    parse_data(message)


def on_error(ws, error):
    print("WS ERROR:", error)


def on_close(ws, close_status_code, close_msg):
    print("WS CLOSED -> reconnect...")
    time.sleep(3)


def on_open(ws):
    print("WS CONNECTED")

    ws.send(json.dumps({
        "M": "EnterLobby",
        "H": "md5luckydiceHub",
        "I": 0
    }))

    def ping():
        while True:
            try:
                ws.send(json.dumps({
                    "M": "PingPong",
                    "H": "md5luckydiceHub",
                    "I": 1
                }))
                time.sleep(5)
            except:
                break

    threading.Thread(target=ping, daemon=True).start()


def start_ws():
    while True:
        try:
            ws = websocket.WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.on_open = on_open
            ws.run_forever()
        except Exception as e:
            print("Reconnect error:", e)

        time.sleep(3)


# ================= API =================
@app.route("/api/taixiumd5")
def api():
    return jsonify({
        "status": "success",
        "data": latest_data
    })


# ================= RUN =================
if __name__ == "__main__":
    threading.Thread(target=start_ws, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
