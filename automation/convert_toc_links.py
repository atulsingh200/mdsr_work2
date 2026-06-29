import re

BASE_URL = "https://experienceleague.adobe.com/en/docs/experience-manager-cloud-service/content"

def convert_url(match):
    path = match.group(1)
    if path.startswith("http://") or path.startswith("https://"):
        return f"]({path})"
    url_path = re.sub(r"^/help", "", path)
    url_path = re.sub(r"\.md$", "", url_path)
    return f"]({BASE_URL}{url_path})"


with open("AEM_TOC.md", "r") as f:
    content = f.read()

# Match only the URL part of markdown links: ]( ... )
converted = re.sub(r"\]\(([^)]+)\)", convert_url, content)

with open("README.md", "w") as f:
    f.write(converted)

remaining = len(re.findall(r"\]\(/help/", converted))
print(f"Done. Written to README.md. Remaining relative links: {remaining}")

lines = [l for l in converted.splitlines() if "experienceleague" in l]
for line in lines[:5]:
    print(line.strip())
