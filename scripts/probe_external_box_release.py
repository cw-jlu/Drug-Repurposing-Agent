"""Read-only, traced availability probe for the public CDRPipe Box release."""

from __future__ import annotations

import json
from pathlib import Path
import re

import requests

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


URL = "https://ucsf.box.com/s/m54ipylmdytjsqmlp7axnabvjh2q8lwl"
OUTPUT = Path("artifacts/reports/external_box_release_probe.json")


def _main(trace: TraceRecorder) -> None:
    response = requests.get(URL, timeout=30)
    trace.emit("shared_page_fetched", url=URL, status=response.status_code,
               bytes=len(response.content), content_type=response.headers.get("Content-Type"))
    response.raise_for_status()
    html = response.text
    names = sorted(set(re.findall(r"[A-Za-z0-9_./-]+\.(?:RData|parquet|csv)", html,
                                  flags=re.IGNORECASE)))
    patterns = {term: len(re.findall(term, html, flags=re.IGNORECASE)) for term in
                ("sharedFolder", "sharedLink", "itemId", "requestToken", "folderId",
                 "api.box.com", "__INITIAL_STATE__", "window.__", "Box.config")}
    scripts = re.findall(r'<script[^>]+src="([^"]+)"', html, flags=re.IGNORECASE)
    snippets = {}
    for term in ("itemId", "folderId", "currentFolderID", "Box.config"):
        match = re.search(term, html, re.IGNORECASE)
        if match:
            snippet = html[max(0, match.start() - 250):match.end() + 120]
            snippet = re.sub(r'(?i)(requestToken.{0,8})[A-Za-z0-9_-]{15,}',
                             r'\1[REDACTED]', snippet)
            snippets[term] = snippet
    config_match = re.search(r"Box\.config\s*=\s*", html)
    config = (json.JSONDecoder().raw_decode(html[config_match.end():])[0]
              if config_match else {})
    def find_folders(value):
        if isinstance(value, dict):
            if "currentFolderID" in value and "items" in value:
                yield {"folder_id": value["currentFolderID"],
                       "folder_name": value.get("currentFolderName"),
                       "items": [{key: item.get(key) for key in ("id", "name", "type", "size")}
                                 for item in value["items"]]}
            for child in value.values():
                yield from find_folders(child)
        elif isinstance(value, list):
            for child in value:
                yield from find_folders(child)
    route_match = re.search(r'"\\/app-api\\/enduserapp\\/shared-folder"\s*:\s*', html)
    route_data = (json.JSONDecoder().raw_decode(html[route_match.end():])[0]
                  if route_match else {})
    initial_folders = list(find_folders(route_data))
    child_folders = []
    for item in (initial_folders[0]["items"] if initial_folders else []):
        if item["type"] != "folder":
            continue
        child_url = f"{URL}/folder/{item['id']}"
        child_response = requests.get(child_url, timeout=30)
        trace.emit("child_folder_page_fetched", url=child_url,
                   status=child_response.status_code, bytes=len(child_response.content))
        if child_response.ok:
            child_match = re.search(r'"\\/app-api\\/enduserapp\\/shared-folder"\s*:\s*',
                                    child_response.text)
            if child_match:
                child_data = json.JSONDecoder().raw_decode(
                    child_response.text[child_match.end():])[0]
                child_folders.extend(find_folders(child_data))
    file_probe = {}
    cmap_matrix = next((item for folder in child_folders for item in folder["items"]
                        if item["name"] == "cmap_signatures.RData"), None)
    if cmap_matrix:
        file_url = f"{URL}/file/{cmap_matrix['id']}"
        file_response = requests.get(file_url, timeout=30)
        trace.emit("matrix_file_page_fetched", url=file_url, status=file_response.status_code,
                   bytes=len(file_response.content))
        file_probe = {"url": file_url, "status": file_response.status_code,
                      "bytes": len(file_response.content),
                      "route_keys": re.findall(r'"(\\/app-api\\/enduserapp\\/[^\"]+)"\s*:',
                                               file_response.text),
                      "size_mentions": re.findall(r'"(?:size|fileSize|sizeBytes)"\s*:\s*\d+',
                                                  file_response.text)[:8]}
        route_details = {}
        for route in (r'\/app-api\/enduserapp\/shared-item',
                      r'\/app-api\/enduserapp\/item\/f_2206971964738'):
            match = re.search('"' + re.escape(route) + r'"\s*:\s*', file_response.text)
            if match:
                data = json.JSONDecoder().raw_decode(file_response.text[match.end():])[0]
                if isinstance(data, dict):
                    route_details[route] = {"keys": sorted(data),
                                            "download_keys": [key for key in data
                                                              if "download" in key.lower() or
                                                              "url" in key.lower()]}
                    if isinstance(data.get("items"), list) and data["items"]:
                        first = data["items"][0]
                        route_details[route]["first_item"] = {
                            "keys": sorted(first),
                            "id": first.get("id"), "name": first.get("name"),
                            "size": first.get("size"),
                            "download_url_present": bool(first.get("downloadURL") or
                                                         first.get("downloadUrl") or
                                                         first.get("download_url"))}
        file_probe["route_details"] = route_details
        direct_url = ("https://ucsf.box.com/index.php?rm=box_download_shared_file"
                      f"&shared_name=m54ipylmdytjsqmlp7axnabvjh2q8lwl&file_id=f_{cmap_matrix['id']}")
        with requests.get(direct_url, stream=True, allow_redirects=False,
                          timeout=30) as download_probe:
            file_probe["direct_download_probe"] = {
                "status": download_probe.status_code,
                "redirect_host": requests.utils.urlparse(
                    download_probe.headers.get("Location", "")).hostname,
                "content_type": download_probe.headers.get("Content-Type"),
                "content_length": download_probe.headers.get("Content-Length"),
                "content_disposition": download_probe.headers.get("Content-Disposition")}
            trace.emit("matrix_download_header_probed", url=direct_url,
                       **file_probe["direct_download_probe"])
            next_url = download_probe.headers.get("Location")
        redirect_steps = []
        for _ in range(3):
            if not next_url or not next_url.startswith("https://"):
                break
            host = requests.utils.urlparse(next_url).hostname
            if host not in {"ucsf.app.box.com", "dl.boxcloud.com"}:
                break
            if host == "dl.boxcloud.com":
                redirect_steps.append({"host": host, "status": "signed_download_url_issued"})
                break
            try:
                with requests.get(next_url, stream=True, allow_redirects=False,
                                  timeout=30) as step:
                    redirect_steps.append({"host": host, "status": step.status_code,
                                           "redirect_host": requests.utils.urlparse(
                                               step.headers.get("Location", "")).hostname})
                    next_url = step.headers.get("Location")
            except requests.RequestException as exc:
                redirect_steps.append({"host": host, "error_type": type(exc).__name__})
                break
        file_probe["redirect_steps"] = redirect_steps
        trace.emit("matrix_redirect_chain_probed", steps=redirect_steps)
    api = requests.get("https://api.box.com/2.0/shared_items",
                       params={"shared_link": URL}, timeout=30)
    trace.emit("shared_api_probed", status=api.status_code, bytes=len(api.content),
               content_type=api.headers.get("Content-Type"))
    payload = {"status": "metadata_probe_only", "url": URL,
               "page_status": response.status_code, "page_bytes": len(response.content),
               "page_title": (re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S).group(1).strip()
                              if re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
                              else None),
               "filename_mentions": names[:100], "api_status": api.status_code,
               "html_markers": patterns, "html_snippets": snippets,
               "config_keys": sorted(config), "initial_folders": initial_folders,
               "child_folders": child_folders,
               "cmap_matrix_file_probe": file_probe,
               "script_sources": scripts[:20],
               "api_preview": api.text[:400], "trace": str(trace.path)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("probe_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               filename_mentions=len(names))
    print(json.dumps({key: payload[key] for key in
                      ("page_status", "page_bytes", "page_title", "filename_mentions",
                       "api_status", "html_markers", "initial_folders",
                       "child_folders", "cmap_matrix_file_probe")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("external_box_release_probe", _main,
               Path("artifacts/reports/traces"))
