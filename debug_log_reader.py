
with open('backend/server_out.log', 'r') as f:
    for line in f:
        if "validation error" in line or "422" in line:
            print(line.strip())
