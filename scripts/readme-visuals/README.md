# README visuals

The wordmark, two learning routes, experiment gallery, and PPO training loop share Instrument Serif, Manrope,
and an indigo accent. English and Chinese figures have light, dark, desktop, and
compact versions. Text is shaped and converted to SVG paths so GitHub does not
need to load fonts.

```bash
python -m pip install fonttools uharfbuzz cairosvg
python scripts/readme-visuals/render.py --preview /tmp/rl-readme-previews
python scripts/readme-visuals/render.py --figure classic-route
python scripts/readme-visuals/render.py --figure modern-route
python scripts/readme-visuals/render.py --figure lab-gallery
```

Sources and SHA-256 hashes are recorded in `fonts.json`. Bundled fonts use the
SIL Open Font License; their license texts are in `fonts/`. Manrope is instantiated
at weight 500. Noto Sans SC is a weight-500 subset containing the figure labels;
add any new Chinese characters to that subset before changing the labels.

The opening routes follow the twenty-six chapters listed in both README editions.
The classical route covers Chapters 1–12, from CartPole and MDPs to value learning,
policy gradients, and control. Its shared-foundations caption connects advantages
and PPO to the modern route. The modern route covers Chapters 13–26, from LLM
post-training to reasoning, tool use, and multimodal environments. Arrows show a
learning progression rather than a single training recipe; RLHF and DPO are
alternative alignment methods. Modern post-training starts with existing models,
not from-scratch pretraining. Each route has pictograms, a shared framework strip,
and a secondary row of related topics. Detailed chapter listings remain in the
README tables. The PPO loop summarizes interaction, rollout collection,
advantage estimation, and policy/value updates; it does not report benchmark
results. The renderer checks glyph coverage, label widths, and text bounds.

The gallery presents eight selected hands-on examples. The first four panels
embed environment frames captured from existing course GIFs; source paths and
hashes are recorded in `frames/sources.json`. The other four panels illustrate
preference pairs, code tests, offline document search, and the GeoQA geometry
task. They depict the tasks rather than training measurements. The Deep Research
entry links to the small offline REINFORCE policy, and GeoQA links to the EasyR1
training walkthrough. Each README includes a link for every example.
