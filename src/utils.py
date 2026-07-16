def sec_to_mmss(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"

def mmss_to_sec(time_str):
    m, s = time_str.split(":")
    return int(m) * 60 + int(s)
