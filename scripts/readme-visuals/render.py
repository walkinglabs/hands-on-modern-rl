#!/usr/bin/env python3
"""Render the bilingual README wordmark and figures as portable SVG outlines.

pip install fonttools uharfbuzz cairosvg
python scripts/readme-visuals/render.py --preview /tmp/rl-readme-previews
"""
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET

import cairosvg
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import BoundsPen

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'docs/public/readme'
COLORS = {
    'light': dict(ink='#171D29', accent='#5654D6', muted='#656C78', line='#DDE1EA', soft='#F5F5FA', bg='#FFFFFF', paper='#FCFCFA'),
    'dark': dict(ink='#F0F3F8', accent='#A5A3FF', muted='#ADB4C1', line='#354050', soft='#1B202C', bg='#0D1117', paper='#0D1117'),
}

COPY = {
    'en': {
        'map': 'Course at a glance', 'total': '7 parts · 26 chapters',
        'foundation': 'Reinforcement learning foundations',
        'modern': 'Language models, agents & multimodal systems',
        'parts': [
            ['I · Chapters 1–4', 'MDPs & value functions', 'CartPole · Bandits', 'Bellman · DP · MC · TD'],
            ['II · Chapters 5–9', 'Deep reinforcement learning', 'DQN · Policy gradients', 'Actor-Critic · PPO · SAC'],
            ['III · Chapters 10–12', 'Data, experts & exploration', 'Offline RL · Imitation learning', 'Exploration · Multi-agent RL'],
            ['IV · Chapters 13–18', 'LLM post-training', 'RLHF · DPO · GRPO · RLVR', 'Reasoning · Process rewards'],
            ['V · Chapters 19–22', 'Agents & tool use', 'Coding · Browser · GUI', 'Multi-turn trajectories'],
            ['VI · Chapters 23–24', 'Multimodal RL', 'VLM · GeoQA · Audio', 'Embodied & visual generation'],
        ],
        'last': ['VII · Chapters 25–26', 'Rewards, evaluation & research', 'Reward hacking · Evaluation · Self-play'],
        'loop': 'The PPO training loop', 'repeat': 'Collect the next batch with the updated policy',
        'stages': [
            ['Interact', 'Policy + environment'],
            ['Collect rollouts', 'Observations · actions · rewards'],
            ['Compute advantages', 'Value estimates · GAE'],
            ['Update the policy', 'PPO clipping · value loss'],
        ],
        'monitor': 'Inspect training',
        'metrics': ['Episode return', 'Policy loss', 'Value loss'],
        'practice': ['Practice.', 'Understand.', 'Build.'],
    },
    'zh': {
        'map': '课程总览', 'total': '七部分 · 二十六章',
        'foundation': '强化学习基础', 'modern': '大模型后训练、智能体与多模态',
        'parts': [
            ['第一部分 · 第 1–4 章', '决策问题与价值函数', 'CartPole · 多臂老虎机', '贝尔曼方程 · DP · MC · TD'],
            ['第二部分 · 第 5–9 章', '深度强化学习', 'DQN · 策略梯度', 'Actor-Critic · PPO · SAC'],
            ['第三部分 · 第 10–12 章', '数据、专家与探索', '离线强化学习 · 模仿学习', '探索 · 多智能体'],
            ['第四部分 · 第 13–18 章', '大模型对齐与后训练', 'RLHF · DPO · GRPO · RLVR', '推理模型 · 过程奖励'],
            ['第五部分 · 第 19–22 章', '智能体与工具调用', '代码 · 浏览器 · 图形界面', '多轮交互与轨迹'],
            ['第六部分 · 第 23–24 章', '多模态强化学习', 'VLM · GeoQA · 音频', '具身智能 · 视觉生成'],
        ],
        'last': ['第七部分 · 第 25–26 章', '奖励、评测与研究前沿', '奖励黑客 · 可靠评测 · 自博弈'],
        'loop': 'PPO 训练循环', 'repeat': '用更新后的策略采集下一批轨迹',
        'stages': [
            ['与环境交互', '策略选择动作，环境返回反馈'],
            ['收集轨迹', '观测 · 动作 · 奖励'],
            ['计算优势', '价值估计 · GAE'],
            ['更新策略', 'PPO 裁剪 · 价值损失'],
        ],
        'monitor': '观察训练指标', 'metrics': ['每回合回报', '策略损失', '价值损失'],
        'practice': ['动手实践', '理解原理', '构建系统'],
    },
}

ICONS = [
    '<rect x="-35" y="-27" width="70" height="52" rx="2"/><path d="M-24 12H24M-12 12L3-18M-12 12V18M12 12V18"/><circle cx="3" cy="-18" r="3"/>',
    '<rect x="-32" y="-28" width="64" height="56" rx="2"/><path d="M-32-10H32M-17-19H-15M-5-19H-3M7-19H9M-19 1H-10M-19 14H-10M0 1H19M0 14H19"/>',
    '<path d="M-31 26V-26M-31 26H32M-21 14L-6 1L7 8L25-20"/><circle cx="-21" cy="14" r="3"/><circle cx="-6" cy="1" r="3"/><circle cx="7" cy="8" r="3"/><circle cx="25" cy="-20" r="3"/>',
    '<path d="M-23-16L0-26L23-16M-23 16L0 26L23 16M-23-16V16M23-16V16M0-26V26M-23-16L23 16M23-16L-23 16"/><circle cx="-23" cy="-16" r="5"/><circle cx="-23" cy="16" r="5"/><circle cx="0" cy="-26" r="5"/><circle cx="0" cy="26" r="5"/><circle cx="23" cy="-16" r="5"/><circle cx="23" cy="16" r="5"/>',
]


class Font:
    def __init__(self, name):
        manifest = json.loads((HERE / 'fonts.json').read_text())
        assert hashlib.sha256((HERE / 'fonts' / name).read_bytes()).hexdigest() == manifest[name]['sha256'], name
        self.tt = TTFont(HERE / 'fonts' / name)
        self.tt.flavor = None
        raw = io.BytesIO(); self.tt.save(raw)
        self.hb = hb.Font(hb.Face(raw.getvalue()))
        self.upem = self.tt['head'].unitsPerEm
        self.hb.scale = (self.upem, self.upem)
        self.glyphs = self.tt.getGlyphSet()
        self.order = self.tt.getGlyphOrder()
        self.cmap = self.tt.getBestCmap()

    @lru_cache(maxsize=None)
    def shape(self, text):
        assert all(ord(c) in self.cmap for c in text), f'Missing glyph in {text!r}'
        buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
        hb.shape(self.hb, buf)
        out, x = [], 0
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            assert info.codepoint != 0
            out.append((info.codepoint, (x + pos.x_offset) / self.upem, pos.y_offset / self.upem))
            x += pos.x_advance
        return out, x / self.upem

    @lru_cache(maxsize=None)
    def outline(self, gid):
        glyph = self.glyphs[self.order[gid]]
        pen = SVGPathPen(self.glyphs, ntos=lambda v: format(v, '.2f').rstrip('0').rstrip('.') if v else '0')
        glyph.draw(pen)
        bp = BoundsPen(self.glyphs); glyph.draw(bp)
        return pen.getCommands(), bp.bounds


class Figure:
    def __init__(self, name, lang, variant, compact, height, fonts, title, description):
        self.name, self.lang, self.variant, self.compact = name, lang, variant, compact
        self.w, self.h = (800 if compact else 1600), height
        self.fonts, self.c, self.bounds, self.outlines = fonts, COLORS[variant], [], {}
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{height}" viewBox="0 0 {self.w} {height}" role="img" aria-labelledby="title desc" lang="{lang}">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
                      f'<rect width="{self.w}" height="{height}" fill="{self.c["paper" if name == "wordmark" else "bg"]}"/>',
                      f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{self.c["muted"]}"/></marker></defs>']

    def text(self, label, x, y, size=30, color='ink', anchor='middle', max_width=None, font=None):
        font = self.fonts[font or ('zh' if self.lang == 'zh' else 'sans')]
        glyphs, width = font.shape(label); width *= size
        assert not max_width or width <= max_width, (label, width, max_width)
        start = x-width/2 if anchor == 'middle' else x-width if anchor == 'end' else x
        self.parts.append(f'<g fill="{self.c[color]}" aria-label="{escape(label)}">')
        for gid, dx, dy in glyphs:
            path, bounds = font.outline(gid)
            if not path: continue
            if path not in self.outlines: self.outlines[path] = f'g{len(self.outlines)}'
            scale, px, py = size/font.upem, start+dx*size, y-dy*size
            self.parts.append(f'<use href="#{self.outlines[path]}" transform="translate({px:.4f} {py:.4f}) scale({scale:.6f} {-scale:.6f})"/>')
            a,b,c,d = bounds
            self.bounds.append((px+a*scale, py-d*scale, px+c*scale, py-b*scale, label))
        self.parts.append('</g>')
        return width

    def lines(self, label, x, y, width, size=34, color='ink', anchor='start'):
        font = self.fonts['zh' if self.lang == 'zh' else 'sans']
        words = list(label) if self.lang == 'zh' else label.split(' ')
        lines, line = [], ''
        for word in words:
            proposed = line + ('' if self.lang == 'zh' or not line else ' ') + word
            if line and font.shape(proposed)[1]*size > width:
                lines.append(line); line = word
            else: line = proposed
        if line: lines.append(line)
        assert len(lines) <= 2, (label, lines)
        for i,line in enumerate(lines): self.text(line,x,y+i*size*1.25,size,color,anchor,width)

    def path(self, d, color='line', width=1.4, arrow=False):
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.c[color]}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def rect(self, x, y, w, h, fill='bg', stroke='line'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{self.c[fill]}" stroke="{self.c[stroke]}" stroke-width="1.4"/>')

    def save(self, preview):
        assert all(x>=20 and y>=18 and u<=self.w-20 and v<=self.h-18 for x,y,u,v,_ in self.bounds), self.bounds
        self.parts.append('<defs>'+''.join(f'<path id="{key}" d="{path}"/>' for path,key in self.outlines.items())+'</defs>')
        source='\n'.join(self.parts+['</svg>'])+'\n'; ET.fromstring(source)
        name = self.name + ('-zh' if self.lang == 'zh' else '') + ('-compact' if self.compact else '') + ('-dark' if self.variant == 'dark' else '')
        OUT.mkdir(parents=True,exist_ok=True); (OUT/f'{name}.svg').write_text(source)
        if preview:
            preview.mkdir(parents=True,exist_ok=True)
            cairosvg.svg2png(bytestring=source.encode(),write_to=str(preview/f'{name}.png'),output_width=800 if self.compact else 960)
        print(name)


def wordmark(lang,variant,compact,fonts,preview):
    t=COPY[lang]; h=460 if compact else 440
    d=Figure('wordmark',lang,variant,compact,h,fonts,'Hands-On Modern RL','Hands-On Modern RL. '+ ' '.join(t['practice']))
    if compact:
        d.text('Hands-On',43,214,170,anchor='start',font='serif',max_width=716)
        end=d.text('Modern',50,326,80,anchor='start',font='sans')
        d.text('RL',50+end+24,326,80,'accent','start',font='sans')
        d.text(' · '.join(s.rstrip('.') for s in t['practice']),50,407,26,'muted','start',max_width=700)
        d.path('M50 437H750'); d.path('M50 437H142','accent',4)
    else:
        d.text('Hands-On',76,233,210,anchor='start',font='serif',max_width=1020)
        end=d.text('Modern',84,360,110,anchor='start',font='sans')
        d.text('RL',84+end+32,360,110,'accent','start',font='sans')
        for i,line in enumerate(t['practice']): d.text(line,1190,245+i*48,33,'accent' if i==2 else 'muted','start',max_width=320)
        d.path('M84 402H1516'); d.path('M84 402H192','accent',5)
    d.save(preview)


def course_map(lang,variant,compact,fonts,preview):
    t=COPY[lang]
    d=Figure('course-map',lang,variant,compact,1920 if compact else 1020,fonts,t['map'],t['total']+'. '+ '; '.join(' · '.join(p) for p in t['parts'])+'; '+' · '.join(t['last']))
    d.text(t['map'],d.w/2,72,42)
    d.text(t['total'],d.w/2,117,24,'muted')
    if compact:
        for i,(meta,title,a,b) in enumerate(t['parts']):
            y=170+i*250
            d.rect(42,y,716,224)
            d.path(f'M42 {y}H116','accent',4)
            d.text(meta,70,y+44,24,'accent','start',max_width=660)
            d.lines(title,70,y+96,658,37)
            d.text(a,70,y+164,27,'muted','start',max_width=658)
            d.text(b,70,y+202,27,'muted','start',max_width=658)
        y=1680
        d.rect(42,y,716,190,'soft')
        d.text(t['last'][0],70,y+43,24,'accent','start')
        d.lines(t['last'][1],70,y+97,658,34)
        d.text(t['last'][2],70,y+157,27,'muted','start',max_width=658)
    else:
        d.text(t['foundation'],70,182,25,'muted','start')
        d.text(t['modern'],70,530,25,'muted','start')
        for i,(meta,title,a,b) in enumerate(t['parts']):
            x=70+(i%3)*498; y=214+(i//3)*348
            d.rect(x,y,464,270)
            d.path(f'M{x} {y}H{x+74}','accent',4)
            d.text(meta,x+27,y+45,23,'accent','start',max_width=410)
            d.lines(title,x+27,y+106,410,34)
            d.text(a,x+27,y+206,25,'muted','start',max_width=410)
            d.text(b,x+27,y+241,25,'muted','start',max_width=410)
        d.rect(70,882,1460,98,'soft')
        d.text(t['last'][0],96,923,23,'accent','start',max_width=435)
        d.text(t['last'][1],550,922,32,anchor='start',max_width=930)
        d.text(t['last'][2],550,958,25,'muted','start',max_width=930)
    d.save(preview)


def training_loop(lang,variant,compact,fonts,preview):
    t=COPY[lang]
    d=Figure('training-loop',lang,variant,compact,1330 if compact else 670,fonts,t['loop'],t['repeat']+'. '+'; '.join(' · '.join(s) for s in t['stages']))
    d.text(t['loop'],d.w/2,73,42)
    if compact:
        for i,(title,sub) in enumerate(t['stages']):
            y=180+i*227
            d.parts.append(f'<g transform="translate(115 {y+30})" fill="none" stroke="{d.c["accent"]}" stroke-width="2.5" stroke-linejoin="round">{ICONS[i]}</g>')
            d.text(f'0{i+1}',186,y+6,24,'accent','start')
            d.text(title,186,y+53,36,anchor='start',max_width=555)
            d.text(sub,186,y+101,25,'muted','start',max_width=555)
            if i<3: d.path(f'M115 {y+90}V{y+161}','muted',1.8,True)
        d.path('M700 919H730V1070H28V210H65','muted',1.8,True)
        d.text(t['repeat'],400,1125,27,'muted',max_width=710)
        d.rect(42,1195,716,81,'soft')
        d.text(' · '.join(t['metrics']),400,1244,25,max_width=686)
    else:
        for i,(title,sub) in enumerate(t['stages']):
            x=250+366*i
            d.text(f'0{i+1}',x,173,24,'accent')
            d.parts.append(f'<g transform="translate({x} 255) scale(1.22)" fill="none" stroke="{d.c["accent"]}" stroke-width="2.2" stroke-linejoin="round">{ICONS[i]}</g>')
            d.text(title,x,354,31,max_width=320)
            d.text(sub,x,398,22,'muted',max_width=335)
            if i<3: d.path(f'M{x+65} 255H{x+293}','muted',1.8,True)
        d.path('M1348 434V500H250V434','muted',1.8,True)
        d.rect(504,477,592,42,'bg','bg')
        d.text(t['repeat'],800,506,25,'muted',max_width=575)
        d.text(t['monitor'],94,592,24,'muted','start')
        for i,metric in enumerate(t['metrics']):
            x=460+i*357
            d.path(f'M{x-55} 569V605')
            d.text(metric,x,592,29,anchor='start',max_width=290)
    d.save(preview)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--preview',type=Path)
    args=parser.parse_args()
    fonts={key:Font(name) for key,name in [('serif','InstrumentSerif-Regular.ttf'),('sans','Manrope-500.ttf'),('zh','NotoSansSC-subset.ttf')]}
    for lang in COPY:
        for variant in COLORS:
            for compact in [False,True]:
                for draw in [wordmark,course_map,training_loop]: draw(lang,variant,compact,fonts,args.preview)
