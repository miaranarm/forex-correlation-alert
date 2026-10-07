import pandas as pd

def returns(series):
    return series.pct_change().dropna()

def correlation_matrix(prices, window=96):
    return returns(prices).tail(window).corr(method="pearson")

def pair_correlation(prices, a, b, window=96):
    x=prices[[a,b]].pct_change().dropna().tail(window)
    return float(x[a].corr(x[b]))

def rolling_correlation(prices, a, b, window=96, short_window=24):
    x=prices[[a,b]].pct_change().dropna()
    if len(x)<window:
        return None
    recent=x.tail(short_window)
    full=x.tail(window)
    return {
        "short":float(recent[a].corr(recent[b])),
        "long":float(full[a].corr(full[b])),
        "spread":float(abs(recent[a].corr(recent[b]))-abs(full[a].corr(full[b])))
    }

def correlation_stability(prices, a, b, window=96, segments=4):
    x=prices[[a,b]].pct_change().dropna().tail(window)
    if len(x)<segments*10:
        return 0.0
    parts=[x.iloc[i*len(x)//segments:(i+1)*len(x)//segments] for i in range(segments)]
    vals=[float(p[a].corr(p[b])) for p in parts if len(p)>2]
    if not vals:
        return 0.0
    same_sign=sum(1 for v in vals if v==0 or (v>0)==(vals[-1]>0))
    magnitude=sum(abs(v) for v in vals)/len(vals)
    return round(0.5*(same_sign/len(vals))+0.5*min(magnitude/0.65,1),4)

def strong_pairs(matrix, threshold=0.65):
    rows=[]
    for i,a in enumerate(matrix.columns):
        for b in matrix.columns[i+1:]:
            r=float(matrix.loc[a,b])
            if abs(r)>=threshold:
                rows.append({"pair_a":a,"pair_b":b,"correlation":r})
    return sorted(rows,key=lambda x:abs(x["correlation"]),reverse=True)
