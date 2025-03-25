import os
import json
from bs4 import BeautifulSoup
import sys
import base64
import time
import traceback
import html2text

text_maker = html2text.HTML2Text()


# pip install html2text
def filter_html(soup):
    convert_br_to_newlines(soup)
    insert_newlines_for_block_elements(soup)
    convert_li_to_newlines(soup)
    text = soup.get_text()
    if text:
        text.strip()
    return text


def convert_br_to_newlines(element):
    for br in element.find_all("br"):
        br.replace_with("\n")


def convert_li_to_newlines(element):
    li_num = 1
    for li in element.find_all("li"):
        if li.text.strip():
            li.append("\n")
            li.insert(0, "{}.".format(li_num))
            li_num += 1


def insert_newlines_for_block_elements(element):
    block_elements = ["p", "pre", "blockquote"]
    for e in element.find_all(block_elements):
        if e.text.strip():
            e.append("\n")


def parse_dir(data_file):
    for name in os.listdir(data_file):
        if name != "result":
            file_name = data_file + "/" + name
            print(file_name)
            with open(file_name) as fd:
                datas = fd.readlines()
            for data in datas:
                item = {"kw": name, "result": []}
                if data.strip():
                    page = json.loads(data.strip())["page"]
                    if page and "Model: GPT-4" in page:
                        soup = BeautifulSoup(page, "html.parser")
                        whitespace = soup.select("div.items-start.gap-4")
                        for wrap in whitespace:
                            content = filter_html(wrap)
                            if "markdown" not in str(wrap):
                                item["result"].append({"question": content})
                            else:
                                filter_html(wrap)
                                item["result"].append({"answer": content})
                if item["result"]:
                    with open("merge_result", "a", encoding="utf-8") as ft:
                        ft.write("{}\n".format(json.dumps(item, ensure_ascii=False)))


def parse_table_to_markdown(table):
    rows = table.find_all("tr")
    markdown_table = ""

    # 解析表头（th）
    header_row = rows[0]
    headers = header_row.find_all("th")
    for header in headers:
        markdown_table += "| " + header.get_text().strip() + " "
    markdown_table += "|\n"

    # 添加表头与表格分隔符
    markdown_table += "|"
    for _ in headers:
        markdown_table += " --- |"
    markdown_table += "\n"

    # 解析表格内容（tr）
    for row in rows[1:]:
        cells = row.find_all("td")
        for cell in cells:
            markdown_table += "| " + cell.get_text().strip() + " "
        markdown_table += "|\n"

    return markdown_table


def parse_poke_parsed_data(ret_json):
    for data in ret_json:
        if "task_id" in data:
            return data
    return None


def parse_json_data(datas):
    json_data = []
    filter_list = ["user", "assistant"]
    for data in datas:
        message = datas[data].get("message")
        if message:
            role = message["author"]["role"]
            if role in filter_list:
                content = message["content"]["parts"][0]
                json_data.append({"role": role, "content": content})
    return json_data


def get_websocket_result(body):
    """
    解析获取ws结果
    """

    payloadData = body["payloadData"]
    if payloadData:
        try:
            payload_data = json.loads(payloadData)
            _type = payload_data.get("type")
            if _type == "message":
                data = payload_data["data"]
                body = base64.b64decode(data["body"]).decode("utf-8")
                return body.replace("data:", "").strip()
        except Exception as err:
            pass
    return ""


def parse_poke_unparsed_data(ret_json, task_id, is_label):
    for d in ret_json:
        if d["type"] == "proxy":
            proxies = d["data"]
            for p in proxies:
                url = p["url"]
                body = p["body"]
                break
    if not url or not body:
        return None

    data = json.loads(body)
    result_body = data["body"]
    status_code = data["status_code"]
    message = data["message"]
    instance_id = data["instance_id"]
    result_data = json.loads(data["data"])
    answer_duration_list = []
    json_data = {}
    after_input = 0
    user_data = ""
    find_page = False
    page = ""
    page_model = "GPT-4"
    for result in result_data:
        if result["type"] == "time":
            if result["tag"] == "after_input":
                after_input = int(result["data"])
            if result["tag"] == "generate_finish":
                answer_duration_list.append(int(result["data"]) - after_input)
        elif result["type"] == "user_data":
            user_data = result["data"]

        elif result["type"] == "page" and result["tag"] == "final":
            page = result["data"]
            page_model = "GPT-4"
            find_page = True
        elif result["type"] == "proxy":
            # 'https://chat.openai.com/backend-api/conversation'
            data = result["data"]
            ws_list = []
            for d in data:
                url = d["url"]
                body = d["body"]
                _type = d["type"]
                if _type == "websocket":
                    data = get_websocket_result(body)
                    try:
                        status = json.loads(data)["message"]["status"]
                        if status == "finished_successfully":
                            json_data = json.loads(data)["message"]
                    except:
                        # 忽略异常
                        pass

    if not find_page:
        page = result_body
        page_model = "GPT-4"
    cost_time = 0
    if len(answer_duration_list) > 0:
        cost_time = answer_duration_list[0]
    chat_list = []

    return {
        "task_id": task_id,
        "status_code": status_code,
        "message": message,
        "page": page,
        "instance_id": instance_id,
        "cost_time": cost_time,
        "is_label": is_label,
        "chat_list": chat_list,
        "answer_duration_list": answer_duration_list,
        "model": "GPT-4",
        "page_model": page_model,
        "json_data": json_data,
        "user_data": user_data,
    }


def get_html(body):
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        "<title>Result</title></head><body>{}</body></html>".format(body)
    )


def parse_old_page(item, text_space):
    is_parse = False
    for wrap in text_space:
        user = wrap.select('div[data-message-author-role="user"]')
        if user:
            content_html = wrap.select("div.text-message")[0]
            item["markdown_result"].append(
                {"question": text_maker.handle(str(content_html))}
            )
            item["html_result"].append({"question": get_html(str(content_html))})
            content = filter_html(content_html)
            item["result"].append({"question": content})
        assistant = wrap.select('div[data-message-author-role="assistant"]')
        if assistant:
            question_html = wrap.select("div.markdown")[0]
            item["html_result"].append({"answer": get_html(str(question_html))})
            item["markdown_result"].append(
                {"answer": text_maker.handle(str(question_html))}
            )
            table = wrap.find("table")
            if table:
                content = parse_table_to_markdown(table)
            else:
                content = filter_html(question_html)
            if content:
                is_parse = True
            item["result"].append({"answer": content})
    return is_parse


def parse_new_page(item, text_space):
    is_parse = False
    for wrap in text_space:
        if "gizmo-bot-avatar" not in str(wrap):
            content_html = wrap.select("div.text-message")[0]
            item["markdown_result"].append(
                {"question": text_maker.handle(str(content_html))}
            )
            item["html_result"].append({"question": get_html(str(content_html))})
            content = filter_html(content_html)
            item["result"].append({"question": content})
            # 提问
        else:
            question_html = wrap.select("div.markdown")[0]
            item["html_result"].append({"answer": get_html(str(question_html))})
            item["markdown_result"].append(
                {"answer": text_maker.handle(str(question_html))}
            )
            table = wrap.find("table")
            if table:
                content = parse_table_to_markdown(table)
            else:
                content = filter_html(question_html)
            if content:
                is_parse = True
            item["result"].append({"answer": content})
            # 回答
    return is_parse


def parse_line(l):
    is_parse = False
    is_label = 0
    data = json.loads(l)
    userdata = data["userdata"]
    result_data = json.loads(data["data"])
    task_id = data.get("task_id", "")
    item = None
    data = parse_poke_parsed_data(result_data)
    if not data:
        data = parse_poke_unparsed_data(result_data, task_id, is_label)
    page = data["page"]
    status_code = data["status_code"]
    user_data = data["user_data"]
    print(f"status_code={status_code}")
    if status_code == 0:
        item = {
            "task_data": userdata,
            "user_data": user_data,
            "json_data": data.get("json_data", []),
            "html_result": [],
            "result": [],
            "markdown_result": [],
            "answer_duration_list": data["answer_duration_list"],
        }

        soup = BeautifulSoup(page, "html.parser")
        text_space = soup.select("article")
        model = "".join([i.text for i in soup.select("span.text-token-text-secondary")])
        if "4o" in model:
            model = "ChatGPT_4o"
        elif "3.5" in model:
            model = "ChatGPT_3.5"
        elif "4" in model:
            model = "ChatGPT_4"
        else:
            model = "unknown"
        item["model"] = model
        if "You said:" in page:
            is_parse = parse_old_page(item, text_space)
        else:
            is_parse = parse_new_page(item, text_space)

    return userdata, item, is_parse


def parse_data(file_name, all_dict):
    i = 0
    parse_num = 0
    result_file = "{}_result.txt".format(file_name.split(".")[0])
    with open(file_name, encoding="utf8") as fd:
        for line in fd:
            try:
                line = line.strip()
                if line:
                    i += 1
                    userdata, item, is_parse = parse_line(line)
                    if item is None:
                        continue
                    if is_parse:
                        parse_num += 1
                    all_dict[userdata] = True
                    with open(result_file, "a", encoding="utf-8") as ft:
                        print(item)
                        ft.write("{}\n".format(json.dumps(item, ensure_ascii=False)))
            except Exception as err:
                traceback.print_exc()
                pass
    print("总行数={},解析成功={},解析成功率={}".format(i, parse_num, parse_num / i))


if __name__ == "__main__":
    md5_query_dict = {}
    all_dict = {}
    parse_data("./parser.txt", all_dict)
