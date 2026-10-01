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

# Course overview pictograms: concrete environments and model interactions.
MAP_ICONS = {
    'cartpole': '<path d="M-48 31H48M-20 1H21V19H-20ZM0 1L21-43"/><circle cx="21" cy="-43" r="4"/><circle cx="-12" cy="24" r="5"/><circle cx="13" cy="24" r="5"/>',
    'mdp': '<rect x="-39" y="-39" width="78" height="78"/><path d="M-13-39V39M13-39V39M-39-13H39M-39 13H39"/><path d="M-26 26V0H0V-26H26M20-32L26-26L20-20"/><circle cx="-26" cy="26" r="4"/>',
    'language': '<path d="M-43-36H20V4H-17L-31 17V4H-43ZM-28-23H4M-28-12H9"/><path d="M-9 13V40H22L36 51V40H46V-4H33M4 26L13 35L30 18"/>',
    'tools': '<rect x="-16" y="-16" width="32" height="32"/><path d="M-7-5L-12 0L-7 5M7-5L12 0L7 5M0-16V-32M-16 0H-32M16 0H35V22"/><rect x="-21" y="-55" width="42" height="23"/><path d="M-21-47H21M-13-51H-11M-5-51H-3"/><rect x="-55" y="-12" width="23" height="25"/><path d="M-49-6L-44-1L-49 4M-42 5H-37M0 16V42H24M24 42L32 34L42 42L34 50ZM29 25L39 35"/>',
    'multimodal': '<rect x="-45" y="-36" width="61" height="46"/><circle cx="-28" cy="-22" r="5"/><path d="M-42 5L-21-13L-8-2L3-11L13-1M1 19H45V-7H27M45 19L32 31V19M11 37L-5 22L-23 34M-23 34V43M-34 43H-12"/><circle cx="-5" cy="22" r="4"/><circle cx="11" cy="37" r="4"/>',
    'data': '<ellipse cx="0" cy="-28" rx="29" ry="10"/><path d="M-29-28V24C-29 38 29 38 29 24V-28M-29-9C-29 5 29 5 29-9M-29 9C-29 23 29 23 29 9"/>',
    'explore': '<circle cx="0" cy="0" r="36"/><path d="M-36 0H-27M27 0H36M0-36V-27M0 27V36M-13 18L-4-4L17-16L7 7Z"/>',
    'check': '<path d="M0-39L30-25V3C30 21 15 33 0 41C-15 33-30 21-30 3V-25ZM-13 0L-3 10L17-12"/>',
    'book': '<path d="M-28-23C-18-27-7-23 0-16C7-23 18-27 28-23V23C18 19 7 23 0 30C-7 23-18 19-28 23ZM0-16V30M-18-10L-7-6M-18 1L-7 5M7-6L18-10M7 5L18 1"/>',
}
MAP_ICONS['network'] = ''.join(f'<path d="M-38 {a}L0 {b}M0 {a}L38 {b}" stroke-width="1.25"/>' for a in [-32,0,32] for b in [-32,0,32]) + ''.join(f'<circle cx="{x}" cy="{y}" r="5" fill="@surface"/>' for x in [-38,0,38] for y in [-32,0,32])
MAP_COPY = {
    'en': {
        'subtitle': 'From CartPole to modern agents',
        'stages': [('Environment','CartPole','cartpole'),('MDPs & values','Bellman','mdp'),('Deep RL','DQN · PPO','network'),('LLM post-training','RLHF · GRPO','language'),('Tool-using agents','Code · Browser','tools'),('Multimodal RL','VLM · Embodied','multimodal')],
        'practice': 'Equations · Code · Experiments',
        'topics': 'Topics across the course',
        'extensions': [('Data & experts','Offline RL · Imitation','data'),('Exploration','Multi-agent · Hierarchy','explore'),('Safety & evaluation','Rewards · Self-play','check')],
    },
    'zh': {
        'subtitle': '从 CartPole 到现代智能体',
        'stages': [('环境交互','CartPole','cartpole'),('MDP 与价值','贝尔曼方程','mdp'),('深度强化学习','DQN · PPO','network'),('大模型后训练','RLHF · GRPO','language'),('工具智能体','代码 · 浏览器','tools'),('多模态强化学习','VLM · 具身智能','multimodal')],
        'practice': '公式推导 · 代码实现 · 动手实验',
        'topics': '贯穿课程的专题',
        'extensions': [('数据与专家','离线强化学习 · 模仿学习','data'),('探索与协作','多智能体 · 分层强化学习','explore'),('安全与评测','奖励设计 · 自博弈','check')],
    },
}


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

    def icon(self, name, x, y, scale=1, color='accent'):
        content=MAP_ICONS[name].replace('@surface',self.c['bg'])
        self.parts.append(f'<g transform="translate({x} {y}) scale({scale})" fill="none" stroke="{self.c[color]}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">{content}</g>')

    def save(self, preview):
        bad=[b for b in self.bounds if b[0]<20 or b[1]<18 or b[2]>self.w-20 or b[3]>self.h-18]
        assert not bad,bad
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
    t=COPY[lang]; visual=MAP_COPY[lang]
    d=Figure('course-map',lang,variant,compact,1210 if compact else 730,fonts,t['map'],t['total']+'. '+ '; '.join(' · '.join(p) for p in t['parts'])+'; '+' · '.join(t['last']))
    d.text('Hands-On Modern RL',d.w/2,62,39,font='sans')
    d.text(visual['subtitle'],d.w/2,106,26,'accent')
    if compact:
        d.rect(42,160,716,728,'bg','accent')
        for i,(title,sub,icon) in enumerate(visual['stages']):
            x=224+(i%2)*352; y=252+(i//2)*250
            d.icon(icon,x,y,1.14)
            d.text(title,x,y+84,28,max_width=320)
            d.text(sub,x,y+123,22,'muted',max_width=320)
            if i%2==0: d.path(f'M{x+76} {y}H{x+276}','muted',1.8,True)
            elif i<5: d.path(f'M{x} {y+146}V{y+173}H224V{y+186}','muted',1.8,True)
        d.rect(42,888,716,38,'accent','accent')
        d.text('Hands-On Modern RL',400,915,24,'bg',font='sans')
        d.path('M400 928V951','muted',1.6,True)
        d.icon('book',147 if lang=='en' else 143,979,.42)
        d.text(visual['practice'],423,986,25,'muted',max_width=556)
        d.text(visual['topics'],400,1035,21,'muted')
        d.rect(42,1054,716,132,'soft','soft')
        for i,(title,sub,icon) in enumerate(visual['extensions']):
            x=164+i*236
            d.icon(icon,x,1087,.46)
            d.text(title,x,1134,22,max_width=230)
            d.text(sub,x,1168,18,'muted',max_width=230)
    else:
        d.rect(220,160,1160,217,'bg','accent')
        positions=[94,365,655,945,1235,1506]
        for i,((title,sub,icon),x) in enumerate(zip(visual['stages'],positions)):
            d.icon(icon,x,252,1.05,'ink' if i in [0,5] else 'accent')
            d.text(title,x,332,(20 if lang=='zh' else 22) if i in [0,5] else 25,max_width=174 if i in [0,5] else 275)
            d.text(sub,x,361,18 if i in [0,5] else 20,'muted',max_width=170 if i in [0,5] else 260)
            if i<5:
                end=positions[i+1]-72 if i not in [0,4] else (207 if i==0 else 1447)
                start=x+72 if i not in [0,4] else (152 if i==0 else 1394)
                d.path(f'M{start} 252H{end}','muted',1.8,True)
        d.rect(220,377,1160,38,'accent','accent')
        d.text('Hands-On Modern RL',800,403,24,'bg',font='sans')
        d.path('M800 417V444','muted',1.6,True)
        d.icon('book',519 if lang=='en' else 562,481,.42)
        d.text(visual['practice'],823,489,23,'muted',max_width=645)
        d.text(visual['topics'],800,542,20,'muted')
        d.rect(60,562,1480,142,'soft','soft')
        for i,(title,sub,icon) in enumerate(visual['extensions']):
            x=307+i*493
            if i: d.path(f'M{x-246} 581V684')
            d.icon(icon,x,597,.49)
            d.text(title,x,646,25,max_width=420)
            d.text(sub,x,680,20,'muted',max_width=440)
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
    parser.add_argument('--figure',choices=['wordmark','course-map','training-loop'])
    args=parser.parse_args()
    fonts={key:Font(name) for key,name in [('serif','InstrumentSerif-Regular.ttf'),('sans','Manrope-500.ttf'),('zh','NotoSansSC-subset.ttf')]}
    for lang in COPY:
        for variant in COLORS:
            for compact in [False,True]:
                figures={'wordmark':wordmark,'course-map':course_map,'training-loop':training_loop}
                for name,draw in figures.items():
                    if not args.figure or args.figure==name: draw(lang,variant,compact,fonts,args.preview)
