#!/usr/bin/env python3
# md_checker.py
"""
Markdown 格式检查器（静态分析）
用法:
    python md_checker.py file.md
    cat file.md | python md_checker.py -
    python md_checker.py file.md --fix   # 尝试自动补闭合的 fenced code block（谨慎）
输出: 控制台报告（也可扩展为 JSON）

功能:
 - 检查未闭合的 fenced code block (``` 或 ~~~)
 - 检查未闭合的 inline code (`...`)
 - 检查表格列数不一致或缺失表头分隔行
 - 检查括号/中括号/花括号不匹配
 - 检查引号总数异常（启发式）
 - 检查链接/图片语法不完整
 - 检测流式分段常见征兆（文件末尾没有换行，以反斜杠结尾等）
"""

import sys
import re
import argparse
from typing import List, Tuple, Dict, Any


# -----------------------
# utils
# -----------------------
def index_to_linecol(text: str, idx: int) -> Tuple[int, int]:
    """将字符索引 idx 转换为 (line_no(1-based), col_no(1-based))"""
    lines = text[:idx].splitlines(keepends=True)
    line_no = len(lines) if lines else 1
    col_no = (len(lines[-1]) + 1) if lines else 1
    if lines:
        col_no = idx - text.rfind('\n', 0, idx)
    return line_no, max(1, col_no)


def mk_issue(typ: str, msg: str, line: int = None, col: int = None, excerpt: str = None) -> Dict[str, Any]:
    return {"type": typ, "message": msg, "line": line, "col": col, "excerpt": excerpt}


# -----------------------
# masking helpers
# -----------------------
def mask_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """用空格替换给定的 [start, end) 跨度，避免在这些区域再做语法检查（例如代码块内部）。"""
    if not spans:
        return text
    arr = list(text)
    for s, e in spans:
        if s < 0: s = 0
        if e > len(arr): e = len(arr)
        for i in range(s, e):
            # 保持换行以便行号映射不变
            if arr[i] != '\n':
                arr[i] = ' '
    return ''.join(arr)


# -----------------------
# checks
# -----------------------
def find_fenced_code_blocks(text: str) -> Tuple[List[Tuple[int, int, str, int]], List[Dict]]:
    """
    返回 (blocks, issues)
    blocks: list of (start_idx, end_idx_or_-1, fence_char, fence_count)
      如果 end_idx_or_-1 == -1 表示未闭合
    issues: any found issues (unclosed)
    """
    issues = []
    lines = text.splitlines(keepends=True)
    blocks = []
    stack = []  # each item: (fence_char, count, start_idx, start_line)
    pos = 0
    fence_re = re.compile(r'^([`~]{3,})(.*)\n?$')
    for i, line in enumerate(lines):
        m = fence_re.match(line)
        if m:
            fence = m.group(1)  # e.g. ``` or ~~~
            ch = fence[0]
            cnt = len(fence)
            if stack and stack[-1][0] == ch:
                # close if current count >= open count (CommonMark rule)
                open_ch, open_cnt, open_pos, open_line = stack[-1]
                if cnt >= open_cnt:
                    stack.pop()
                    blocks.append((open_pos, pos + len(line), ch, open_cnt))
                else:
                    # treat as inner fence (rare)
                    stack.append((ch, cnt, pos, i + 1))
            else:
                # open
                stack.append((ch, cnt, pos, i + 1))
        pos += len(line)
    # any remaining opens are unclosed
    for open_ch, open_cnt, open_pos, open_line in stack:
        blocks.append((open_pos, -1, open_ch, open_cnt))
        issues.append(mk_issue(
            "fenced_code_unclosed",
            f"Fenced code block opened with '{open_ch * open_cnt}' at line {open_line} is not closed.",
            line=open_line,
            col=1,
            excerpt=text[open_pos:open_pos + 80].splitlines()[0] if open_pos < len(text) else None
        ))
    return blocks, issues


def check_inline_code_unmatched(text: str) -> List[Dict]:
    """
    使用正则先找到所有成对的 inline code (`...`、``...``、等)，把它们替换掉，
    然后检测剩余的反引号是否存在（说明有未闭合）。
    """
    issues = []
    # find matched inline code spans
    # pattern: (`+)(.*?)\1  NON-GREEDY
    matches = list(re.finditer(r'(`+)(.+?)\1', text, flags=re.S))
    spans = [(m.start(), m.end()) for m in matches]
    masked = mask_spans(text, spans)
    # if any backtick remains, likely unmatched
    if '`' in masked:
        idx = masked.find('`')
        line, col = index_to_linecol(text, idx)
        issues.append(mk_issue(
            "inline_code_unmatched",
            "发现未闭合的内联代码反引号（`）。",
            line=line, col=col,
            excerpt=text[idx:idx + 60].splitlines()[0]
        ))
    return issues


def check_brackets_balance(text: str) -> List[Dict]:
    """
    基于字符栈检查 (), [], {} 的平衡。会忽略被 fenced code block 或 inline code 包围的区域。
    """
    issues = []
    # mask code spans to avoid false positives
    blocks, _ = find_fenced_code_blocks(text)
    spans = []
    for s, e, ch, c in blocks:
        if e > 0:
            spans.append((s, e))
        else:
            spans.append((s, len(text)))
    # mask inline code spans (to avoid `)`)
    inline_matches = list(re.finditer(r'(`+)(.+?)\1', text, flags=re.S))
    for m in inline_matches:
        spans.append((m.start(), m.end()))
    masked = mask_spans(text, spans)
    pair = {'(': ')', '[': ']', '{': '}'}
    openers = set(pair.keys())
    closers = {v: k for k, v in pair.items()}
    stack = []
    for i, ch in enumerate(masked):
        if ch in openers:
            stack.append((ch, i))
        elif ch in closers:
            if not stack:
                line, col = index_to_linecol(text, i)
                issues.append(mk_issue("bracket_unmatched", f"发现多余的闭合符号 '{ch}'。", line=line, col=col,
                                       excerpt=text[i:i + 40]))
            else:
                top, pos = stack[-1]
                if pair[top] == ch:
                    stack.pop()
                else:
                    # mismatch
                    line, col = index_to_linecol(text, i)
                    issues.append(
                        mk_issue("bracket_mismatch", f"闭合符号 '{ch}' 与打开符号 '{top}' 不匹配。", line=line, col=col,
                                 excerpt=text[i:i + 40]))
                    stack.pop()
    for opener, pos in stack:
        line, col = index_to_linecol(text, pos)
        issues.append(mk_issue("bracket_unclosed", f"打开的符号 '{opener}' 在文档中没有对应闭合。", line=line, col=col,
                               excerpt=text[pos:pos + 40]))
    return issues


def check_quotes(text: str) -> List[Dict]:
    """
    启发式检查引号是否成对：统计 ASCII 单/双引号和中文左右引号。
    注意：存在误报（例如缩写 don't），只是提示。
    """
    issues = []
    # mask code fragments to avoid false positives
    blocks, _ = find_fenced_code_blocks(text)
    spans = [(s, e if e > 0 else len(text)) for s, e, _, _ in blocks]
    inline_matches = list(re.finditer(r'(`+)(.+?)\1', text, flags=re.S))
    for m in inline_matches:
        spans.append((m.start(), m.end()))
    masked = mask_spans(text, spans)  # text

    # 2. 定义引号对
    quote_pairs = {'“': '”', '‘': '’', '"': '"', "'": "'"}
    left_quotes = {'“', '‘', '"', "'"}
    right_quotes = {'”', '’', '"', "'"}

    # 3. 逐行扫描匹配
    lines = masked.splitlines()
    for lineno, line in enumerate(lines, 1):
        stack = []
        for col, ch in enumerate(line, 1):
            if ch not in left_quotes and ch not in right_quotes:
                continue
            # 对称引号 ASCII
            if ch in ['"', "'"]:
                if stack and stack[-1][0] == ch:
                    stack.pop()
                else:
                    stack.append((ch, col))
            # 左引号
            elif ch in ['“', '‘']:
                stack.append((ch, col))
            # 右引号
            elif ch in ['”', '’']:
                if stack and quote_pairs.get(stack[-1][0]) == ch:
                    stack.pop()
                else:
                    issues.append(mk_issue(
                        "quote_unmatched",
                        f"发现未闭合的右引号 {ch}",
                        lineno, col,
                        line.strip()
                    ))
        # 栈里剩下的就是未闭合左引号
        for ch, col in stack:
            issues.append(mk_issue(
                "quote_unmatched",
                f"发现未闭合的左引号 {ch}",
                lineno, col,
                line.strip()
            ))

    return issues



def check_tables(text: str) -> List[Dict]:
    """
    检测 markdown 表格的不一致：
    - 找到连续包含 '|' 的行块
    - 检查表格前是否有空行
    - 检查表头分隔行是否存在
    - 检查每行的列数是否一致
    """
    issues = []
    lines = text.splitlines()
    n = len(lines)
    i = 0
    table_sep_re = re.compile(r'^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$')
    while i < n:
        if '|' in lines[i]:
            # gather block
            j = i
            block = []
            while j < n and ('|' in lines[j] or lines[j].strip() == ''):
                # allow blank lines within small gaps
                if lines[j].strip() == '' and (j + 1 < n and '|' not in lines[j + 1]):
                    break
                block.append((j, lines[j]))
                j += 1

            # candidate table block must have at least 2 pipe lines
            pipe_lines = [ln for _, ln in block if '|' in ln]
            if len(pipe_lines) >= 2:
                header_idx, header_line = block[0]

                # # 检查表格前空行
                # if header_idx > 0 and lines[header_idx - 1].strip() != '':
                #     issues.append({
                #         "type": "table_no_blank_line_before",
                #         "message": f"表格前应有空行 (行 {header_idx + 1})",
                #         "line": header_idx + 1,
                #         "col": 1,
                #         "excerpt": header_line.strip()
                #     })

                # 检查表头分隔行
                if len(block) >= 2 and table_sep_re.match(block[1][1]):
                    pass
                else:
                    issues.append({
                        "type": "table_missing_separator",
                        "message": f"表格在行 {header_idx + 1} 开始，但缺失表头分隔行 (如 '| --- | --- |')",
                        "line": header_idx + 1,
                        "col": 1,
                        "excerpt": header_line.strip()
                    })

                # 列数检查
                def col_count(s):
                    parts = [p.strip() for p in s.strip().strip('|').split('|')]
                    return len(parts)

                expected = col_count(header_line)
                for idx, ln in block:
                    if '|' not in ln:
                        continue
                    c = col_count(ln)
                    if c != expected:
                        issues.append({
                            "type": "table_col_mismatch",
                            "message": f"表格行 {idx + 1} 列数 ({c}) 与表头列数 ({expected}) 不一致",
                            "line": idx + 1,
                            "col": 1,
                            "excerpt": ln.strip()
                        })
            i = j
        else:
            i += 1

    return issues


def check_links_images(text: str) -> List[Dict]:
    """
    检查常见的链接/图片语法不闭合情况，例如 [text](url  或 ![alt](url
    也会依赖于通用括号检查
    """
    issues = []
    # look for patterns of '[' without matching ']' or '(' without matching ')'
    # quick heuristics: find occurrences of '[' followed later by '(' on the same line
    lines = text.splitlines()
    for lineno, ln in enumerate(lines, start=1):
        # if there's '[' but no ']' in line -> suspicious
        if '[' in ln and ']' not in ln:
            issues.append(mk_issue("link_syntax", "行中存在 '[' 但没有 ']'，可能是未完成的链接或文本。", line=lineno,
                                   col=ln.find('[') + 1, excerpt=ln.strip()))
        if '(' in ln and ')' not in ln and re.search(r'\[[^\]]*\]\s*\(', ln):
            # pattern [text]( but missing )
            issues.append(mk_issue("link_syntax", "发现可能不完整的链接/图片语法（缺失右括号 ')'）。", line=lineno,
                                   col=ln.find('(') + 1, excerpt=ln.strip()))
    return issues


def check_streaming_signs(text: str, strict: bool = False) -> List[Dict]:
    """
    针对流式返回常见的迹象提供提示：
     - 可选：文档最后一行没有换行（strict=True 时才检查）
     - 文档以不完整的一行结束（如以反斜杠结尾）
     - 文档末尾以未闭合的标记结尾（比如以单个 '[' 或 '`' 结尾）
    """
    issues = []
    lines = text.splitlines()
    last_line = lines[-1] if lines else ''

    # 1. 仅在 strict 模式下检查末行是否换行
    if strict and not text.endswith('\n'):
        issues.append(
            mk_issue("stream_incomplete", "文档最后一行没有换行（可能是流式输出被截断）。",
                     line=len(lines), col=len(last_line) + 1))

    # 2. 检查末行是否有明显未完成迹象
    if last_line.endswith('\\'):
        issues.append(
            mk_issue("stream_incomplete",
                     "最后一行以反斜杠结尾，可能是行被截断或转义未完成。",
                     line=len(lines), col=len(last_line)))

    stripped = last_line.strip()
    if stripped and stripped[-1] in ['[', '(', '`']:
        issues.append(
            mk_issue("stream_incomplete",
                     "最后一行以可能需要闭合的符号结尾（例如 '[', '(', '`'），可能被截断。",
                     line=len(lines), col=len(last_line)))

    return issues


# -----------------------
# orchestrator
# -----------------------
def run_all_checks(text: str) -> List[Dict]:
    issues = []
    # 1. fenced code blocks
    _, fenced_issues = find_fenced_code_blocks(text)
    issues.extend(fenced_issues)
    # 2. inline code
    issues.extend(check_inline_code_unmatched(text))
    # 3. brackets
    issues.extend(check_brackets_balance(text))
    # 4. quotes
    issues.extend(check_quotes(text))
    # 5. tables
    issues.extend(check_tables(text))
    # 6. links/images
    issues.extend(check_links_images(text))
    # 7. streaming signs
    issues.extend(check_streaming_signs(text))
    return issues


# -----------------------
# simple fixer (only for fenced code)
# -----------------------
def fix_fenced_code_blocks(text: str) -> Tuple[str, List[str]]:
    """
    如果存在未闭合的 fenced block，自动在文档末尾追加对应的闭合 fence。
    返回 (new_text, list_of_actions)
    """
    actions = []
    blocks, _ = find_fenced_code_blocks(text)
    unclosed = [b for b in blocks if b[1] == -1]
    if not unclosed:
        return text, actions
    # for each unclosed, append a closing fence of the same char and count (>=3)
    for start, _, ch, cnt in unclosed:
        fence = ch * max(3, cnt)
        actions.append(f"Append closing fence '{fence}' for block opened at byte offset {start}.")
        text += ("\n" if not text.endswith("\n") else "") + fence + "\n"
    return text, actions


# -----------------------
# CLI
# -----------------------
def print_report(issues: List[Dict]):
    if not issues:
        print("No issues found.")
        return
    print(f"Found {len(issues)} issue(s):\n")
    for idx, it in enumerate(issues, start=1):
        loc = f" (line {it['line']}, col {it['col']})" if it.get('line') else ""
        print(f"{idx}. [{it['type']}] {it['message']}{loc}")
        if it.get('excerpt'):
            print("    >", it['excerpt'].strip())
        print()


def main(argv):
    ap = argparse.ArgumentParser(description="Markdown格式检查器")
    ap.add_argument("file", help="要检查的文件路径，使用 - 表示从 stdin 读取")
    ap.add_argument("--fix", action="store_true", help="尝试自动修复部分问题（仅闭合 fenced code block）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出问题（简易）")
    args = ap.parse_args(argv)

    if args.file == "-":
        text = sys.stdin.read()
        source = "<stdin>"
    else:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        source = args.file

    if args.fix:
        text, actions = fix_fenced_code_blocks(text)
        if actions:
            print("Autofix actions applied:")
            for a in actions:
                print(" -", a)
            # write back to file if not stdin
            if args.file != "-":
                with open(args.file, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Patched file: {args.file}")
            else:
                print("Input was stdin; patched text printed below.\n")
                print(text)

    issues = run_all_checks(text)
    if args.json:
        import json
        print(json.dumps(issues, ensure_ascii=False, indent=2))
    else:
        print(f"Report for: {source}\n")
        print_report(issues)


if __name__ == "__main__":
    main(sys.argv[1:])  # python md_checker.py README.md --fix --json:python script/md_checker.py ./README.md
    # issues = check_quotes(
    #     '''基于工商登记信息与年报数据的交叉验证分析，贵州冰阳汽车租赁有限公司存在以下需关注的经营迹象：\n\n**核心风险点：**\n1.  **财务健康状况堪忧**：2021年度年报显示公司已处于**资不抵债**状态，**所有者权益为-13.05万元**。尽管当年有1.25万元的营业收入和0.55万元的净利润，但**负债总额（30.14万元）远超资产总额（17.09万元）**，表明公司存在显著的偿债风险。\n2.  **资本实缴情况异常**：工商登记与历年年报均显示股东认缴出资额为**10万元**，但**实缴出资额持续为0元**。值得注意的是，其实缴期限在年报中多次变更（如从2030年变更为2065年，再变为2055年），这种长期认缴且无实缴资金到位的情况，与汽车租赁行业通常需要一定资产规模的特点不甚匹配。\n3.  **经营规模与人员波动**：公司**当前参保人数为0人**，与2022年及2020年年报披露的**3名从业人员**存在明显矛盾，可能暗示业务收缩或用工模式变化。联系方式（电话、邮箱）及邮政编码在近几年频繁变更，也反映出经营稳定性不足。\n\n**需关注的异常迹象：**\n*   **信息一致性存疑**：工商登记的认缴出资日期（2049年）与年报信息（2055年）存在不一致，虽可能为填报差异，但仍需留意。\n*   **“司法案件”企业标签**：企业标签中包含“司法案件”，提示存在涉诉记录，需结合具体案件情况评估其业务合规性与潜在债务纠纷风险。\n*   **业务资质匹配性**：经营范围包含“旅行社业务（持证经营）”，需关注其是否实际取得并维持相关业务资质。\n\n**综合分析结论：**\n该公司作为一家存续的小微企业，最突出的风险在于2021年披露的**资不抵债**的财务状况，结合**实缴资本为零、人员信息矛盾**以及联系方式频繁变更等情况，综合判断其持续经营能力存在较大不确定性。建议在业务合作中重点关注其当前的实际运营状态、资产情况与债务风险。''')
    # print(issues)
