import requests
import time
import threading
import math
from flask import Flask, jsonify
from collections import defaultdict, deque
from datetime import datetime

# ===============================
# CẤU HÌNH
# ===============================
API_URL   = "https://wtxmd52.tele68.com/v1/txmd5/sessions"
MAX_PHIEN = 20
MIN_PHIEN = 15

app = Flask(__name__)

# ===============================
# DATA
# ===============================
history  = deque(maxlen=MAX_PHIEN)
hist_pt  = deque(maxlen=MAX_PHIEN)
lich_su  = []
stats    = {"tong":0,"dung":0,"sai":0,"cd":0,"cs":0,"max_cd":0,"max_cs":0}
last_phien  = None
_prev_pred  = None

latest_data = {
    "Phiên":          None,
    "Xúc xắc 1":      None,
    "Xúc xắc 2":      None,
    "Xúc xắc 3":      None,
    "Tổng":           None,
    "Kết":            None,
    "Phiên hiện tại": None,
    "Dự đoán":        "Chưa đủ dữ liệu",
    "Độ tin cậy":     0,
    "Pattern":        "",
    "ID":             "tuananh"
}

# ═══════════════════════════════════════════
#  AI ENGINE – 15 MÔ HÌNH
# ═══════════════════════════════════════════

_t1 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t2 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t3 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t4 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_t5 = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_ng = defaultdict(lambda: {"Tài":0,"Xỉu":0})
_sd = {"Tài": defaultdict(int), "Xỉu": defaultdict(int)}
_acc = {k: {"ok":0,"n":0} for k in
        ("m1","m2","m3","m4","m5","ng","sk","pt","fr10","fr20","mom","rep","alt","cyc","bay")}
_prev_model = {}


def _train_markov():
    for tb in (_t1,_t2,_t3,_t4,_t5):
        for d in tb.values(): d.update({"Tài":0,"Xỉu":0})
    h=list(history)
    for i in range(len(h)-1): _t1[h[i]][h[i+1]] += 1
    for i in range(len(h)-2): _t2[h[i]+"|"+h[i+1]][h[i+2]] += 1
    for i in range(len(h)-3): _t3[h[i]+"|"+h[i+1]+"|"+h[i+2]][h[i+3]] += 1
    for i in range(len(h)-4): _t4[h[i]+"|"+h[i+1]+"|"+h[i+2]+"|"+h[i+3]][h[i+4]] += 1
    for i in range(len(h)-5): _t5["|".join(h[i:i+5])][h[i+5]] += 1

def _s(table, key):
    d=table.get(key,{"Tài":0,"Xỉu":0}); t=d["Tài"]+d["Xỉu"]
    if not t: return {"Tài":0.0,"Xỉu":0.0}
    return {"Tài":d["Tài"]/t,"Xỉu":d["Xỉu"]/t}

def _sc_markov():
    h=list(history)
    s1=_s(_t1,h[-1])            if len(h)>=1 else {"Tài":0.0,"Xỉu":0.0}
    s2=_s(_t2,h[-2]+"|"+h[-1])  if len(h)>=2 else {"Tài":0.0,"Xỉu":0.0}
    s3=_s(_t3,"|".join(h[-3:])) if len(h)>=3 else {"Tài":0.0,"Xỉu":0.0}
    s4=_s(_t4,"|".join(h[-4:])) if len(h)>=4 else {"Tài":0.0,"Xỉu":0.0}
    s5=_s(_t5,"|".join(h[-5:])) if len(h)>=5 else {"Tài":0.0,"Xỉu":0.0}
    return s1,s2,s3,s4,s5

def _train_ngram():
    _ng.clear(); h=list(history)
    for ln in range(1,13):
        for i in range(len(h)-ln):
            _ng["|".join(h[i:i+ln])][h[i+ln]] += 1

def _sc_ngram():
    sc={"Tài":0.0,"Xỉu":0.0}; h=list(history)
    for ln in range(min(12,len(h)),0,-1):
        pat="|".join(h[-ln:]); d=_ng.get(pat)
        if not d: continue
        t=d["Tài"]+d["Xỉu"]
        if not t: continue
        w=ln**4; sc["Tài"]+=w*d["Tài"]/t; sc["Xỉu"]+=w*d["Xỉu"]/t
    return sc

def _train_streak():
    for d in _sd.values(): d.clear()
    h=list(history)
    if not h: return
    cur,cnt=h[0],1
    for r in h[1:]:
        if r==cur: cnt+=1
        else: _sd[cur][cnt]+=1; cur,cnt=r,1
    _sd[cur][cnt]+=1

def _cur_streak():
    h=list(history)
    if not h: return None,0
    cur=h[-1]; cnt=1
    for r in reversed(h[:-1]):
        if r==cur: cnt+=1
        else: break
    return cur,cnt

def _sc_streak():
    cur,ln=_cur_streak()
    if not cur: return {"Tài":0.5,"Xỉu":0.5}
    dist=_sd[cur]
    ended =sum(v*(k**1.5) for k,v in dist.items() if k<=ln)
    longer=sum(v*(k**1.5) for k,v in dist.items() if k>ln)
    total=ended+longer
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    other="Xỉu" if cur=="Tài" else "Tài"
    return {cur:longer/total, other:ended/total}

def _sc_point(w=20):
    pts=list(hist_pt)
    if len(pts)<5: return {"Tài":0.5,"Xỉu":0.5}
    recent=pts[-w:]
    avg=sum(recent)/len(recent)
    n=len(recent)
    if n>=6:
        h1=sum(recent[:n//2])/(n//2)
        h2=sum(recent[n//2:])/(n-n//2)
        slope=(h2-h1)/10.5
    else: slope=0
    p_t=max(0.0,min(1.0,(avg-3)/15+slope*0.1))
    return {"Tài":p_t,"Xỉu":1-p_t}

def _sc_freq(w):
    h=list(history)
    if len(h)<w: return {"Tài":0.5,"Xỉu":0.5}
    ct=h[-w:].count("Tài"); p_x=ct/w
    return {"Tài":1-p_x,"Xỉu":p_x}

def _sc_momentum():
    h=list(history)
    if len(h)<20: return {"Tài":0.5,"Xỉu":0.5}
    p5 =h[-5:].count("Tài")/5  if len(h)>=5  else 0.5
    p10=h[-10:].count("Tài")/10 if len(h)>=10 else 0.5
    p20=h[-20:].count("Tài")/20 if len(h)>=20 else 0.5
    mom=(p5-p10)*0.6+(p10-p20)*0.4
    return {"Tài":max(0.0,min(1.0,0.5+mom)),"Xỉu":max(0.0,min(1.0,0.5-mom))}

def _sc_repeat():
    h=list(history)
    if len(h)<8: return {"Tài":0.5,"Xỉu":0.5}
    sc={"Tài":0.0,"Xỉu":0.0}
    for cycle in (2,3,4,5):
        if len(h)<cycle*3: continue
        match=sum(1 for i in range(1,4) if h[-i]==h[-i-cycle])
        if match>=2:
            w=match*cycle; sc[h[-cycle]]+=w
    total=sc["Tài"]+sc["Xỉu"]
    if not total: return {"Tài":0.5,"Xỉu":0.5}
    return {"Tài":sc["Tài"]/total,"Xỉu":sc["Xỉu"]/total}

def _sc_alternating():
    h=list(history)
    if len(h)<6: return {"Tài":0.5,"Xỉu":0.5}
    recent=h[-8:]
    switches=sum(1 for i in range(len(recent)-1) if recent[i]!=recent[i+1])
    if switches>=6:
        other="Xỉu" if h[-1]=="Tài" else "Tài"
        return {other:0.72, h[-1]:0.28}
    if switches<=1:
        return {h[-1]:0.70, ("Xỉu" if h[-1]=="Tài" else "Tài"):0.30}
    return {"Tài":0.5,"Xỉu":0.5}

def _sc_cycle():
    h=list(history)
    if len(h)<12: return {"Tài":0.5,"Xỉu":0.5}
    best_score=0; best_pred=None
    for cycle in range(2,7):
        if len(h)<cycle*2+1: continue
        match=sum(1 for i in range(len(h)-cycle) if h[i]==h[i+cycle])
        total=len(h)-cycle
        if total==0: continue
        score=match/total
        if score>best_score and score>0.7:
            best_score=score
            idx=len(h)%cycle
            if idx==0: idx=cycle
            best_pred=h[-idx] if idx<=len(h) else None
    if best_pred:
        return {best_pred:best_score, ("Xỉu" if best_pred=="Tài" else "Tài"):1-best_score}
    return {"Tài":0.5,"Xỉu":0.5}

def _sc_bayesian():
    h=list(history)
    if len(h)<8: return {"Tài":0.5,"Xỉu":0.5}
    log_odds=0.0
    w5=h[-5:].count("Tài")/5 if len(h)>=5 else 0.5
    if w5>0.8:   log_odds -= 1.2
    elif w5<0.2: log_odds += 1.2
    cur,ln=_cur_streak()
    if cur and ln>=4:
        log_odds += -0.7*ln if cur=="Tài" else 0.7*ln
    if len(h)>=4:
        alt=sum(1 for i in range(3) if h[-1-i]!=h[-2-i])
        if alt==3: log_odds += 0.5 if h[-1]=="Xỉu" else -0.5
    prob=1/(1+math.exp(-log_odds))
    return {"Tài":prob,"Xỉu":1-prob}

def _entropy(w=30):
    h=list(history)[-w:]; n=len(h)
    if n==0: return 1.0
    ct=h.count("Tài"); cx=n-ct
    if ct==0 or cx==0: return 0.0
    pt,px=ct/n,cx/n
    return -(pt*math.log2(pt)+px*math.log2(px))

def _aw(key, base):
    a=_acc[key]
    if a["n"]<15: return base
    return max(0.005, base*(1+4.0*(a["ok"]/a["n"]-0.5)))

def _win(sc):
    return "Tài" if sc.get("Tài",0)>=sc.get("Xỉu",0) else "Xỉu"

def _update_acc(actual):
    for k,p in _prev_model.items():
        if p: _acc[k]["n"]+=1; _acc[k]["ok"]+=(p==actual)

def _update_stats(actual, phien_id=None):
    global _prev_pred
    if not _prev_pred or _prev_pred=="Chưa đủ dữ liệu": return
    dung=(_prev_pred==actual)
    stats["tong"]+=1
    if dung:
        stats["dung"]+=1; stats["cd"]+=1; stats["cs"]=0
        if stats["cd"]>stats["max_cd"]: stats["max_cd"]=stats["cd"]
    else:
        stats["sai"]+=1; stats["cs"]+=1; stats["cd"]=0
        if stats["cs"]>stats["max_cs"]: stats["max_cs"]=stats["cs"]

def _acc_str(key):
    a=_acc[key]
    if a["n"]==0: return "Chưa có"
    return f"{a['ok']}/{a['n']} ({a['ok']/a['n']*100:.0f}%)"


# ===============================
# DỰ ĐOÁN – 15 MÔ HÌNH
# ===============================
def predict():
    if len(history)<MIN_PHIEN:
        return "Chưa đủ dữ liệu", 0

    _train_markov(); _train_ngram(); _train_streak()
    e=_entropy()
    s1,s2,s3,s4,s5=_sc_markov()
    sng=_sc_ngram(); ssk=_sc_streak(); spt=_sc_point()
    sf10=_sc_freq(10); sf20=_sc_freq(20)
    smom=_sc_momentum(); srep=_sc_repeat()
    salt=_sc_alternating(); scyc=_sc_cycle(); sbay=_sc_bayesian()
    ef=max(0.3,1-e*0.4)

    w1=_aw("m1",0.05); w2=_aw("m2",0.07); w3=_aw("m3",0.10)
    w4=_aw("m4",0.11); w5=_aw("m5",0.11); wng=_aw("ng",0.15*ef)
    wsk=_aw("sk",0.08); wpt=_aw("pt",0.05); wf10=_aw("fr10",0.04)
    wf20=_aw("fr20",0.04); wmom=_aw("mom",0.04); wrep=_aw("rep",0.04)
    walt=_aw("alt",0.04); wcyc=_aw("cyc",0.04); wbay=_aw("bay",0.04)
    tw=w1+w2+w3+w4+w5+wng+wsk+wpt+wf10+wf20+wmom+wrep+walt+wcyc+wbay

    raw={}
    for r in ("Tài","Xỉu"):
        raw[r]=(w1*s1.get(r,0)+w2*s2.get(r,0)+w3*s3.get(r,0)+w4*s4.get(r,0)+
                w5*s5.get(r,0)+wng*sng.get(r,0)+wsk*ssk.get(r,0)+wpt*spt.get(r,0)+
                wf10*sf10.get(r,0)+wf20*sf20.get(r,0)+wmom*smom.get(r,0)+
                wrep*srep.get(r,0)+walt*salt.get(r,0)+wcyc*scyc.get(r,0)+
                wbay*sbay.get(r,0))/tw

    s=raw["Tài"]+raw["Xỉu"]
    if s>0: raw={r:v/s for r,v in raw.items()}
    else:   raw={"Tài":0.5,"Xỉu":0.5}

    pred="Tài" if raw["Tài"]>=raw["Xỉu"] else "Xỉu"
    conf=max(raw["Tài"],raw["Xỉu"])

    counted=[_acc[k]["ok"]/_acc[k]["n"] for k in _acc if _acc[k]["n"]>=10]
    hist_acc=sum(counted)/len(counted) if counted else 0.5
    all_p=[_win(s1),_win(s2),_win(s3),_win(s4),_win(s5),_win(sng),_win(ssk),
           _win(spt),_win(sf10),_win(sf20),_win(smom),_win(srep),
           _win(salt),_win(scyc),_win(sbay)]
    dong_thuan=all_p.count(pred)/len(all_p)
    raw_conf=(conf-0.5)*2
    acc_bonus=max(0,hist_acc-0.5)*2
    thuan_bonus=max(0,dong_thuan-0.5)*2
    score=raw_conf*0.50+acc_bonus*0.30+thuan_bonus*0.20
    tin_cay=round(max(50.0,min(100.0,50+score*50)),1)

    global _prev_model
    _prev_model={
        "m1":_win(s1),"m2":_win(s2),"m3":_win(s3),"m4":_win(s4),"m5":_win(s5),
        "ng":_win(sng),"sk":_win(ssk),"pt":_win(spt),"fr10":_win(sf10),
        "fr20":_win(sf20),"mom":_win(smom),"rep":_win(srep),
        "alt":_win(salt),"cyc":_win(scyc),"bay":_win(sbay)
    }
    return pred, tin_cay


# ===============================
# FETCH DATA LOOP
# ===============================
def fetch_loop():
    global last_phien, latest_data, _prev_pred

    while True:
        try:
            res  = requests.get(API_URL, timeout=5)
            data = res.json()
            ds   = data.get("list", [])
            if not ds: time.sleep(2); continue

            item  = ds[0]
            phien = item.get("id")
            if phien==last_phien: time.sleep(2); continue

            d1,d2,d3 = item.get("dices")
            tong      = item.get("point")
            ket       = "Tài" if tong>=11 else "Xỉu"

            # Cập nhật stats & accuracy
            _update_stats(ket, phien)
            if len(history)>=MIN_PHIEN: _update_acc(ket)

            # Lưu lịch sử
            history.append(ket)
            hist_pt.append(tong)

            lich_su.append({
                "phien":   phien,
                "ket_qua": ket,
                "tong":    tong
            })
            if len(lich_su)>200: lich_su.pop(0)

            # Pattern
            pattern = "".join(
                ["T" if x=="Tài" else "X" for x in list(history)[-20:]]
            )

            # Dự đoán
            du_doan, do_tin_cay = predict()
            _prev_pred = du_doan

            so=len(history); td=stats["tong"]
            acc_s=f"{stats['dung']}/{td} ({stats['dung']/td*100:.1f}%)" if td else "Chưa có"

            latest_data = {
                "Phiên":          phien,
                "Xúc xắc 1":      d1,
                "Xúc xắc 2":      d2,
                "Xúc xắc 3":      d3,
                "Tổng":           tong,
                "Kết":            ket,
                "Phiên hiện tại": phien+1,
                "Dự đoán":        du_doan,
                "Độ tin cậy":     do_tin_cay,
                "Pattern":        pattern,
                "ID":             "tuananh"
            }

            # Log giữ nguyên format gốc
            if du_doan=="Chưa đủ dữ liệu":
                print(f"[{phien}] {d1}-{d2}-{d3} | {tong} | {ket} | Chờ {MIN_PHIEN-so} phiên...")
            else:
                print(f"[{phien}] {d1}-{d2}-{d3} | {tong} | {ket} | >>> {du_doan} <<< ({do_tin_cay}%) | {acc_s}")

            last_phien = phien

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Lỗi:", e)

        time.sleep(2)


# ===============================
# THREAD START
# ===============================
threading.Thread(target=fetch_loop, daemon=True).start()


# ===============================
# API
# ===============================
@app.route("/api/taixiumd5", methods=["GET"])
def api_data():
    return jsonify({"data": latest_data})

@app.route("/api/lichsu", methods=["GET"])
def api_lichsu():
    td=stats["tong"]
    return jsonify({
        "tong":       len(lich_su),
        "du_doan":    td,
        "dung":       stats["dung"],
        "sai":        stats["sai"],
        "ty_le":      f"{stats['dung']/td*100:.1f}%" if td else "0%",
        "max_cd":     stats["max_cd"],
        "max_cs":     stats["max_cs"],
        "lich_su":    lich_su[-50:]
    })


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
