import os, time
import numpy as np, pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

app = Flask(__name__, static_folder="static")
S = {}
NUM = ["cgpa", "backlogs", "num_projects", "internships", "num_certifications", "aptitude_score"]

def load(df):
    sk = [c for c in df.columns if c.startswith("skill_")]
    if not sk or "student_id" not in df: raise ValueError("CSV needs student_id and skill_* columns")
    for c in sk + [n for n in NUM if n in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())
    for n in NUM:
        if n not in df: df[n] = 0
    S.update(df=df.reset_index(drop=True), sk=sk, X=StandardScaler().fit_transform(df[sk]))
    S["xy"] = PCA(2, random_state=0).fit_transform(S["X"])

def fit_algo(algo, k, X):
    if algo == "kmeans": return KMeans(k, n_init=10, random_state=42).fit_predict(X)
    if algo == "hierarchical": return AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    if algo == "gmm": return GaussianMixture(k, random_state=42).fit_predict(X)
    d = NearestNeighbors(n_neighbors=8).fit(X).kneighbors(X)[0][:, -1]
    return DBSCAN(eps=float(np.percentile(d, 85)), min_samples=8).fit_predict(X)

def metrics(X, l):
    m = l != -1
    if len(set(l[m])) < 2: return None
    r = dict(silhouette=silhouette_score(X[m], l[m]), davies_bouldin=davies_bouldin_score(X[m], l[m]),
             calinski=calinski_harabasz_score(X[m], l[m]), clusters=len(set(l[m])), noise=int((~m).sum()))
    if "true_profile" in S["df"]: r["ari"] = adjusted_rand_score(S["df"].true_profile, l)
    return {a: round(float(b), 3) for a, b in r.items()}

def summarize(l):
    df, sk, out = S["df"], S["sk"], []
    for c in sorted(set(l)):
        g = df[l == c]; m = g[sk].mean().sort_values(ascending=False)
        top = [s.replace("skill_", "").replace("_", " ") for s in m.index[:3]]
        out.append(dict(id=int(c), size=len(g), name=" + ".join(top[:2]) if c != -1 else "Noise / Outliers",
                        top=[f"{t} ({m.iloc[i]:.1f})" for i, t in enumerate(top)],
                        cgpa=round(float(g.cgpa.mean()), 2),
                        centroid={s: float(v) for s, v in g[sk].mean().items()}))
    return out

@app.route("/")
def home(): return send_from_directory("static", "index.html")

@app.get("/api/students")
def api_students():
    df = S.get("df")
    if df is None: return jsonify(students=[], total=0)
    query = request.args.get("q", "").lower().strip()
    branch = request.args.get("branch", "").strip()
    filtered = df
    if branch and branch != "all":
        filtered = filtered[filtered["branch"] == branch]
    if query:
        name_match = filtered["name"].astype(str).str.lower().str.contains(query, na=False)
        id_match = filtered["student_id"].astype(str).str.lower().str.contains(query, na=False)
        filtered = filtered[name_match | id_match]

    limit = int(request.args.get("limit", 60))
    offset = int(request.args.get("offset", 0))
    total = len(filtered)
    rows = []
    sk = S.get("sk", [])
    for _, r in filtered.iloc[offset:offset+limit].iterrows():
        top_s = sorted([(s.replace("skill_", "").replace("_", " "), float(r[s])) for s in sk if s in r], key=lambda x: x[1], reverse=True)[:3]
        rows.append({
            "student_id": str(r.student_id),
            "name": str(r.get("name", "Student")),
            "branch": str(r.get("branch", "N/A")),
            "cgpa": float(r.get("cgpa", 0.0)),
            "backlogs": int(r.get("backlogs", 0)),
            "num_projects": int(r.get("num_projects", 0)),
            "internships": int(r.get("internships", 0)),
            "aptitude_score": float(r.get("aptitude_score", 0.0)),
            "cluster": int(r.cluster) if "cluster" in r else None,
            "top_skills": [f"{s} ({int(v)})" for s, v in top_s if v > 0]
        })
    return jsonify(students=rows, total=total)

@app.post("/api/add_student")
def api_add_student():
    j = request.json or {}
    df = S.get("df")
    if df is None: return jsonify(error="Dataset not loaded"), 400

    student_id = j.get("student_id") or f"STU{len(df)+1:04d}"
    name = j.get("name") or "New Candidate"
    branch = j.get("branch") or "CSE"
    cgpa = float(j.get("cgpa", 7.0))
    backlogs = int(j.get("backlogs", 0))
    num_projects = int(j.get("num_projects", 1))
    internships = int(j.get("internships", 0))
    aptitude_score = float(j.get("aptitude_score", 65.0))

    new_row = {
        "student_id": student_id,
        "name": name,
        "branch": branch,
        "cgpa": cgpa,
        "backlogs": backlogs,
        "num_projects": num_projects,
        "internships": internships,
        "num_certifications": int(j.get("num_certifications", 1)),
        "aptitude_score": aptitude_score
    }
    for s in S["sk"]:
        clean = s.replace("skill_", "")
        new_row[s] = float(j.get(s, j.get(clean, 2)))

    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    load(new_df)
    return jsonify(success=True, student_id=student_id, message="Student profile registered successfully", total=len(new_df))

@app.post("/api/load")
def api_load():
    try:
        f = request.files.get("file")
        load(pd.read_csv(f) if f else pd.read_csv("data/students.csv"))
    except Exception as e: return jsonify(error=str(e)), 400
    df = S["df"]
    return jsonify(n=len(df), skills=[s.replace("skill_", "") for s in S["sk"]],
                   branches=df.branch.value_counts().to_dict() if "branch" in df else {},
                   cgpa=round(float(df.cgpa.mean()), 2), has_truth="true_profile" in df)

@app.post("/api/cluster")
def api_cluster():
    j = request.json; l = fit_algo(j.get("algo", "kmeans"), int(j.get("k", 8)), S["X"]); S["labels"] = l
    S["df"]["cluster"] = l
    pts = [[round(float(x), 3), round(float(y), 3), int(c), S["df"].student_id[i]] for i, ((x, y), c) in enumerate(zip(S["xy"], l))]
    return jsonify(points=pts, clusters=summarize(l), metrics=metrics(S["X"], l))

@app.post("/api/compare")
def api_compare():
    k, res = int(request.json.get("k", 8)), []
    for a in ["kmeans", "hierarchical", "gmm", "dbscan"]:
        t = time.time(); l = fit_algo(a, k, S["X"]); res.append(dict(algo=a, time=round(time.time() - t, 3), **(metrics(S["X"], l) or {})))
    scan = []
    for kk in range(2, 11):
        km = KMeans(kk, n_init=5, random_state=42).fit(S["X"])
        scan.append(dict(k=kk, inertia=round(float(km.inertia_)), silhouette=round(float(silhouette_score(S["X"], km.labels_)), 3)))
    return jsonify(results=res, scan=scan)

import re, json

def extract_job_data(req_obj):
    data = {}
    f = req_obj.files.get("file") if hasattr(req_obj, "files") else None
    if f:
        content = f.read().decode("utf-8", errors="ignore")
        try: data = json.loads(content)
        except Exception: data = {"raw_text": content}
    elif req_obj.is_json:
        data = req_obj.get_json() or {}
    else:
        raw = req_obj.form.get("json_text") or req_obj.form.get("text") or ""
        try: data = json.loads(raw)
        except Exception: data = {"raw_text": raw}

    if isinstance(data, str):
        try: data = json.loads(data)
        except Exception: data = {"raw_text": data}

    known_skills = {s.replace("skill_", "").lower(): s for s in S.get("sk", [])}
    req_skills = {}

    raw_skills = data.get("skills", {})
    if isinstance(raw_skills, dict):
        for k, v in raw_skills.items():
            clean_k = k.replace("skill_", "").lower()
            orig = known_skills.get(clean_k, f"skill_{k}")
            try:
                val = int(v)
                if val > 0: req_skills[orig] = val
            except: pass
    elif isinstance(raw_skills, list):
        for item in raw_skills:
            if isinstance(item, str):
                clean_k = item.replace("skill_", "").lower()
                orig = known_skills.get(clean_k, f"skill_{item}")
                req_skills[orig] = 3
            elif isinstance(item, dict):
                k = item.get("name") or item.get("skill")
                lvl = item.get("level") or item.get("proficiency") or 3
                if k:
                    clean_k = str(k).replace("skill_", "").lower()
                    orig = known_skills.get(clean_k, f"skill_{k}")
                    req_skills[orig] = int(lvl)

    raw_text = data.get("raw_text", "")
    if not req_skills and raw_text:
        text_lower = raw_text.lower()
        for clean_k, col in known_skills.items():
            pattern = r'\b' + re.escape(clean_k.replace('_', ' ')) + r'\b|\b' + re.escape(clean_k) + r'\b'
            if re.search(pattern, text_lower):
                req_skills[col] = 3

    min_cgpa = data.get("min_cgpa")
    if min_cgpa is None and raw_text:
        m = re.search(r'(?:cgpa|gpa)[\s:=]+([0-9.]+)', raw_text, re.IGNORECASE) or re.search(r'([0-9.]+)\s*(?:\+)?\s*cgpa', raw_text, re.IGNORECASE)
        if m: min_cgpa = float(m.group(1))
    min_cgpa = float(min_cgpa) if min_cgpa is not None else 6.0

    max_backlogs = data.get("max_backlogs")
    if max_backlogs is None and raw_text:
        if re.search(r'no\s+backlog|zero\s+backlog|0\s+backlog', raw_text, re.IGNORECASE):
            max_backlogs = 0
        else:
            m = re.search(r'backlog[s]?[\s:=]+([0-9]+)', raw_text, re.IGNORECASE)
            if m: max_backlogs = int(m.group(1))
    max_backlogs = int(max_backlogs) if max_backlogs is not None else 0

    return {
        "job_title": data.get("job_title") or data.get("role") or data.get("title") or "Custom Job Profile",
        "company": data.get("company") or "Hiring Company",
        "min_cgpa": min_cgpa,
        "max_backlogs": max_backlogs,
        "relevant_only": bool(data.get("relevant_only", data.get("relevant_clusters_only", True))),
        "top": int(data.get("top", 100)),
        "skills": req_skills
    }

@app.post("/api/match")
def api_match():
    parsed = extract_job_data(request)
    req = parsed["skills"]
    df = S["df"]
    if "labels" not in S: S["labels"] = fit_algo("kmeans", 8, S["X"]); df["cluster"] = S["labels"]
    if not req:
        return jsonify(error="No matching skills found in the provided JSON or file. Please ensure skills are specified."), 400

    fit = np.mean([np.minimum(df[c] / v, 1) for c, v in req.items()], axis=0)
    exp = np.minimum(df.num_projects / 4, 1) * .5 + np.minimum(df.internships / 2, 1) * .3 + np.minimum(df.num_certifications / 3, 1) * .2
    df["score"] = 100 * (.65 * fit + .15 * df.cgpa / 10 + .10 * exp + .10 * df.aptitude_score / 100)
    cl = summarize(S["labels"]); cf = {}
    for c in cl: cf[c["id"]] = float(np.mean([min(c["centroid"][k] / v, 1) for k, v in req.items()]))
    best = max(cf.values()) if cf.values() else 1.0
    rel = {c for c, v in cf.items() if v >= .75 * best and c != -1}
    ok = (df.cgpa >= parsed["min_cgpa"]) & (df.backlogs <= parsed["max_backlogs"])
    if parsed["relevant_only"]: ok &= df.cluster.isin(rel)
    rows = []
    for _, r in df[ok].sort_values("score", ascending=False).head(parsed["top"]).iterrows():
        have = [k.replace("skill_", "").replace("_", " ") for k, v in req.items() if r[k] >= v]
        miss = [dict(skill=k.replace("skill_", "").replace("_", " "), has=int(r[k]), need=v) for k, v in req.items() if r[k] < v]
        cn = next((c["name"] for c in cl if c["id"] == r.cluster), "Unknown")
        why = f"Meets {len(have)}/{len(req)} required skills" + (f" ({', '.join(have[:4])})" if have else "") + \
              f"; CGPA {r.cgpa:.1f}; {int(r.num_projects)} projects, {int(r.internships)} internships; cluster: {cn}."
        rows.append(dict(id=r.student_id, name=r.get("name", ""), branch=r.get("branch", ""), cgpa=float(r.cgpa), score=round(float(r.score), 1),
                         cluster=int(r.cluster), cluster_name=cn, missing=miss, why=why))

    detected = {k.replace("skill_", ""): v for k, v in req.items()}
    return jsonify(
        rows=rows, eligible=int(ok.sum()), total=len(df),
        job_info=dict(title=parsed["job_title"], company=parsed["company"], min_cgpa=parsed["min_cgpa"],
                      max_backlogs=parsed["max_backlogs"], required_skills=detected),
        clusters=[dict(id=c["id"], name=c["name"], fit=round(cf[c["id"]] * 100), relevant=c["id"] in rel) for c in cl]
    )

if os.path.exists("data/students.csv"): load(pd.read_csv("data/students.csv"))
if __name__ == "__main__": app.run(debug=True)
