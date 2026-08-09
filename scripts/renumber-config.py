#!/usr/bin/env python3
"""Update config.mjs sidebar/nav numbers to match new chapter numbering."""

import re

CONFIG_PATH = '/Users/sanbu/Code/2026重要开源项目/hands-on-modern-rl/docs/.vitepress/config.mjs'

CHAPTER_MAP = {old: old - 2 for old in range(7, 33)}

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# Pattern 1: Sidebar text labels like "7. Deep Q-Networks", "8. Policy Gradient Methods", etc.
# Match: text: 'N. Title' where N >= 7
def replace_sidebar_label(match):
    old = int(match.group(1))
    new = CHAPTER_MAP.get(old)
    if new is None:
        return match.group(0)
    return f"text: '{new}. {match.group(2)}'"

content = re.sub(r"text: '(\d+)\. ([^']+)'", replace_sidebar_label, content)

# Pattern 2: Section labels like "7.1 From Q-Learning", "10.3 GAE", etc.
def replace_section_label(match):
    old_chap = int(match.group(1))
    new_chap = CHAPTER_MAP.get(old_chap)
    if new_chap is None:
        return match.group(0)
    return f"text: '{new_chap}.{match.group(2)} {match.group(3)}'"

content = re.sub(r"text: '(\d+)\.(\d+) ([^']+)'", replace_section_label, content)

# Pattern 3: 第 X 章 in body text/comments
def replace_chapter_ref(match):
    old = int(match.group(1))
    new = CHAPTER_MAP.get(old)
    if new is None:
        return match.group(0)
    return f'第 {new} 章'

content = re.sub(r'第 (\d+) 章', replace_chapter_ref, content)

if content != original:
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print('config.mjs updated')
else:
    print('config.mjs unchanged')
