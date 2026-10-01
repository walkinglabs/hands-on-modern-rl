# README visuals

The wordmark, course map, and PPO training loop share Instrument Serif, Manrope,
and an indigo accent. English and Chinese figures have light, dark, desktop, and
compact versions. Text is shaped and converted to SVG paths so GitHub does not
need to load fonts.

```bash
python -m pip install fonttools uharfbuzz cairosvg
python scripts/readme-visuals/render.py --preview /tmp/rl-readme-previews
```

Sources and SHA-256 hashes are recorded in `fonts.json`. Bundled fonts use the
SIL Open Font License; their license texts are in `fonts/`. Manrope is instantiated
at weight 500. Noto Sans SC is a weight-500 subset containing the figure labels;
add any new Chinese characters to that subset before changing the labels.

Course labels follow the seven parts and twenty-six chapters listed in both
README editions. The PPO loop summarizes interaction, rollout collection,
advantage estimation, and policy/value updates; it does not report benchmark
results. The renderer checks glyph coverage, label widths, and text bounds.
