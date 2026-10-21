/* global process */
import { defineConfig } from 'vitepress'
import { createLogger } from 'vite'
import { MermaidMarkdown } from 'vitepress-plugin-mermaid'
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const require = createRequire(import.meta.url)
const markdownItFootnote = require('markdown-it-footnote')
const markdownItContainer = require('markdown-it-container')
const katex = require('katex')

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const packageJsonPath = path.resolve(__dirname, '../../package.json')
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
const docsRoot = path.resolve(__dirname, '..')
const assetManifestPath = path.resolve(
  docsRoot,
  'public/optimized/asset-manifest.json'
)

function loadAssetManifest() {
  if (!fs.existsSync(assetManifestPath)) return { assets: {} }

  try {
    return JSON.parse(fs.readFileSync(assetManifestPath, 'utf8'))
  } catch {
    return { assets: {} }
  }
}

const assetManifest = loadAssetManifest()

const isVercel = process.env.VERCEL === '1' || !!process.env.VERCEL_URL

function parseRepository() {
  const repositoryUrl =
    process.env.GITHUB_REPOSITORY ||
    packageJson.repository?.url ||
    packageJson.repository ||
    ''

  if (repositoryUrl.includes('/')) {
    if (repositoryUrl.includes(':')) {
      const sshMatch = repositoryUrl.match(/github\.com:(.+?)\/(.+?)(\.git)?$/)
      if (sshMatch) {
        return { owner: sshMatch[1], repo: sshMatch[2] }
      }
    }

    const normalized = repositoryUrl
      .replace(/^https?:\/\/github\.com\//, '')
      .replace(/^git@github\.com:/, '')
      .replace(/\.git$/, '')

    if (normalized.includes('/')) {
      const [owner, repo] = normalized.split('/')
      return { owner, repo }
    }
  }

  return { owner: 'walkinglabs', repo: packageJson.name || 'course-template' }
}

const { owner, repo } = parseRepository()
const base = process.env.BASE || (isVercel ? '/' : `/${repo}/`)
const siteUrl = process.env.SITE_URL || `https://${owner}.github.io/${repo}`
const editLinkPattern = `https://github.com/${owner}/${repo}/edit/main/docs/:path`
const enableLocalSearch = process.env.LOCAL_SEARCH === '1'
const mermaidConfig = {
  securityLevel: 'loose',
  startOnLoad: false
}

function mermaidConfigPlugin() {
  const virtualModuleId = 'virtual:mermaid-config'
  const resolvedVirtualModuleId = `\0${virtualModuleId}`

  return {
    name: 'local-mermaid-config',
    resolveId(id) {
      if (id === virtualModuleId) {
        return resolvedVirtualModuleId
      }
    },
    load(id) {
      if (id === resolvedVirtualModuleId) {
        return `export default ${JSON.stringify(mermaidConfig)}`
      }
    }
  }
}

function normalizeBrokenDocPathPlugin() {
  const canonicalSegments = ['appendix_math', 'linear-algebra-basics']

  return {
    name: 'normalize-broken-doc-path',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next()

        const url = new URL(req.url, 'http://localhost')
        if (!url.pathname.includes('appendix')) return next()

        const decodedPathname = decodeURIComponent(url.pathname)
        const normalizedDecodedPathname = decodedPathname
          .replace(/\s+/g, '')
          .replace(/appendix_m+ath/gi, canonicalSegments[0])
          .replace(/linea+r-algebra-basics/gi, canonicalSegments[1])

        if (normalizedDecodedPathname !== decodedPathname) {
          const normalizedPathname = normalizedDecodedPathname
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/')
          const redirectTarget = `${normalizedPathname}${url.search}`
          res.statusCode = 302
          res.setHeader('Location', redirectTarget)
          res.end()
          return
        }

        next()
      })
    }
  }
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function slugifySearchHeading(value) {
  return value
    .normalize('NFKD')
    .replace(/[\u0300-\u036F]/g, '')
    .replace(/[\x00-\x1f]/g, '')
    .replace(/[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/]+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function stripMarkdown(value) {
  return value
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_~>#-]/g, '')
    .trim()
}

function renderSearchMarkdown(src) {
  const html = []
  const slugCounts = new Map()
  let inFence = false

  for (const rawLine of src.replace(/^---[\s\S]*?---\n/, '').split('\n')) {
    const line = rawLine.trim()

    if (line.startsWith('```')) {
      inFence = !inFence
      continue
    }

    if (!line || inFence || line.startsWith(':::')) {
      continue
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line)
    if (heading) {
      const level = heading[1].length
      const title = stripMarkdown(heading[2])
      const escapedTitle = escapeHtml(title)
      const baseSlug = slugifySearchHeading(title)
      const count = slugCounts.get(baseSlug) || 0
      slugCounts.set(baseSlug, count + 1)
      const slug = count === 0 ? baseSlug : `${baseSlug}-${count}`
      html.push(
        `<h${level}>${escapedTitle}<a class="header-anchor" href="#${slug}"></a></h${level}>`
      )
      continue
    }

    html.push(`<p>${escapeHtml(stripMarkdown(line))}</p>`)
  }

  return html.join('\n')
}

function isValidMathDelimiter(state, pos) {
  const max = state.posMax
  const prevChar = pos > 0 ? state.src.charCodeAt(pos - 1) : -1
  const nextChar = pos + 1 < max ? state.src.charCodeAt(pos + 1) : -1

  return {
    canOpen: nextChar !== 0x20 && nextChar !== 0x09,
    canClose:
      prevChar !== 0x20 &&
      prevChar !== 0x09 &&
      (nextChar < 0x30 || nextChar > 0x39)
  }
}

function mathInline(state, silent) {
  if (state.src[state.pos] !== '$') return false

  // Display math inline: $$...$$ inside a paragraph/list/blockquote line.
  // Must be tried before the single-$ branch so the closing $$ isn't read as
  // two consecutive empty inline-math delimiters.
  if (state.pos + 1 < state.posMax && state.src[state.pos + 1] === '$') {
    const start = state.pos + 2
    const close = state.src.indexOf('$$', start)
    if (close !== -1 && close < state.posMax) {
      if (!silent) {
        const token = state.push('math_inline', 'math', 0)
        token.markup = '$$'
        token.content = state.src.slice(start, close)
        token.displayMode = true
      }
      state.pos = close + 2
      return true
    }
  }

  let delimiter = isValidMathDelimiter(state, state.pos)
  if (!delimiter.canOpen) {
    if (!silent) state.pending += '$'
    state.pos += 1
    return true
  }

  const start = state.pos + 1
  const max = state.posMax
  let match = start
  while ((match = state.src.indexOf('$', match)) !== -1) {
    if (match >= max) {
      match = -1
      break
    }
    let pos = match - 1
    while (state.src[pos] === '\\') pos -= 1
    if ((match - pos) % 2 === 1) break
    match += 1
  }

  if (match === -1) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (match - start === 0) {
    if (!silent) state.pending += '$$'
    state.pos = start + 1
    return true
  }

  delimiter = isValidMathDelimiter(state, match)
  if (!delimiter.canClose) {
    if (!silent) state.pending += '$'
    state.pos = start
    return true
  }

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(start, match)
  }

  state.pos = match + 1
  return true
}

function mathBlock(state, start, end, silent) {
  let pos = state.bMarks[start] + state.tShift[start]
  const max = state.eMarks[start]

  if (pos + 2 > max) return false
  if (state.src.slice(pos, pos + 2) !== '$$') return false

  pos += 2
  let firstLine = state.src.slice(pos, max)
  let lastLine = ''
  let found = false
  let next = start

  if (silent) return true
  if (firstLine.trim().slice(-2) === '$$') {
    firstLine = firstLine.trim().slice(0, -2)
    found = true
  }

  while (!found) {
    next++
    if (next >= end) break

    pos = state.bMarks[next] + state.tShift[next]
    const lineMax = state.eMarks[next]
    if (pos < lineMax && state.tShift[next] < state.blkIndent) break

    if (state.src.slice(pos, lineMax).trim().slice(-2) === '$$') {
      const lastPos = state.src.slice(0, lineMax).lastIndexOf('$$')
      lastLine = state.src.slice(pos, lastPos)
      found = true
    }
  }

  state.line = next + 1
  const token = state.push('math_block', 'math', 0)
  token.block = true
  token.content =
    (firstLine && firstLine.trim() ? `${firstLine}\n` : '') +
    state.getLines(start + 1, next, state.tShift[start], true) +
    (lastLine && lastLine.trim() ? lastLine : '')
  token.map = [start, state.line]
  token.markup = '$$'
  return true
}

function renderKatex(content, displayMode) {
  try {
    return katex.renderToString(content, {
      displayMode,
      output: 'html',
      throwOnError: true, // Enable error throwing for debugging
      strict: false,
      trust: true
    })
  } catch (error) {
    // Log detailed error information with file context
    const fs = require('fs')
    const path = require('path')
    const markdownFile = path.join(
      process.cwd(),
      'docs/chapter10_ppo/ppo-math.md'
    )

    console.error('\n' + '='.repeat(80))
    console.error('❌ KaTeX Rendering Error')
    console.error('='.repeat(80))
    console.error(
      `Mode: ${displayMode ? 'Display Math ($$...$$)' : 'Inline Math ($...$)'}`
    )
    console.error(
      `Expression: ${content.substring(0, 200)}${content.length > 200 ? '...' : ''}`
    )
    console.error(`Error: ${error.message}`)
    if (error.position !== undefined) {
      console.error(`Position in expression: ${error.position}`)
      const start = Math.max(0, error.position - 50)
      const end = Math.min(content.length, error.position + 50)
      console.error(`Context: ...${content.substring(start, end)}...`)
    }

    // Try to find this expression in the markdown file
    try {
      const mdContent = fs.readFileSync(markdownFile, 'utf8')
      const searchStr = content.substring(0, Math.min(100, content.length))
      const index = mdContent.indexOf(searchStr)
      if (index !== -1) {
        const lineNum = mdContent.substring(0, index).split('\n').length
        console.error(
          `Location: ${markdownFile}, approximately line ${lineNum}`
        )

        // Show surrounding lines
        const lines = mdContent.split('\n')
        const startLine = Math.max(0, lineNum - 3)
        const endLine = Math.min(lines.length, lineNum + 2)
        console.error('\nSurrounding context:')
        for (let i = startLine; i < endLine; i++) {
          const marker = i + 1 === lineNum ? '>>> ' : '    '
          console.error(`${marker}${i + 1}: ${lines[i].substring(0, 100)}`)
        }
      }
    } catch (e) {
      // Ignore file reading errors
    }

    console.error('='.repeat(80) + '\n')

    // Return error HTML for visibility in the page
    return `<div style="background: #fee; border: 2px solid #c00; padding: 10px; margin: 10px 0; border-radius: 4px;">
      <strong style="color: #c00;">KaTeX Rendering Error</strong>
      <p style="margin: 5px 0;"><strong>Mode:</strong> ${displayMode ? 'Display' : 'Inline'}</p>
      <p style="margin: 5px 0;"><strong>Expression:</strong> <code>${content.substring(0, 150)}${content.length > 150 ? '...' : ''}</code></p>
      <p style="margin: 5px 0;"><strong>Error:</strong> ${error.message}</p>
      <p style="margin: 5px 0; font-size: 0.9em; color: #666;">Check browser console for detailed stack trace and file location.</p>
    </div>`
  }
}

function rescueMathInInline(md) {
  // Phase 1 (core, before inline): protect $...$ from emphasis/strong rules.
  // Replace inline-math in raw token content with unique placeholders so that
  // markdown's emphasis rule never sees the underscores inside formulas.
  let mathCounter = 0
  const mathStore = new Map()

  md.core.ruler.before('inline', 'math_inline_protect', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline') continue
      if (!token.content.includes('$')) continue

      token.content = token.content.replace(
        /\$([^\$]+)\$/g,
        function (match, formula) {
          const key = '\x01MATH' + mathCounter++ + '\x01'
          mathStore.set(key, formula)
          return key
        }
      )
    }
  })

  // Phase 2 (core, after inline): restore placeholders as math_inline tokens.
  md.core.ruler.after('inline', 'math_inline_restore', function (state) {
    for (const token of state.tokens) {
      if (token.type !== 'inline' || !token.children) continue

      let hasPlaceholder = false
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          hasPlaceholder = true
          break
        }
      }
      if (!hasPlaceholder) continue

      const newChildren = []
      for (const child of token.children) {
        if (child.type === 'text' && child.content.includes('\x01')) {
          const parts = child.content.split(/(\x01MATH\d+\x01)/)
          for (const part of parts) {
            if (!part) continue
            if (part.startsWith('\x01MATH') && mathStore.has(part)) {
              const t = new state.Token('math_inline', 'math', 0)
              t.content = mathStore.get(part)
              t.markup = '$'
              newChildren.push(t)
            } else if (part.trim()) {
              const t = new state.Token('text', '', 0)
              t.content = part
              newChildren.push(t)
            }
          }
        } else {
          newChildren.push(child)
        }
      }
      token.children = newChildren
    }
  })
}

function katexMarkdown(md) {
  md.inline.ruler.push('math_inline', mathInline)
  md.block.ruler.after('blockquote', 'math_block', mathBlock, {
    alt: ['paragraph', 'reference', 'blockquote', 'list']
  })
  md.renderer.rules.math_inline = (tokens, idx) =>
    renderKatex(tokens[idx].content, tokens[idx].displayMode || false)
  md.renderer.rules.math_block = (tokens, idx) =>
    `<p>${renderKatex(tokens[idx].content, true)}</p>\n`
}

function footnoteTitlePlugin(md) {
  md.renderer.rules.footnote_block_open = (tokens, idx, options, env) => {
    const previousContentToken = [...tokens]
      .slice(0, idx)
      .reverse()
      .find((token) => token.type === 'inline' && token.content?.trim())
    const previousContent = (previousContentToken?.content || '')
      .replace(/[*_`#]/g, '')
      .trim()
    const hasManualTitle = /^(参考文献|References)[:：]?$/.test(previousContent)
    const title = env.relativePath?.startsWith('en/')
      ? 'References'
      : '参考文献'
    const heading = hasManualTitle
      ? ''
      : `<div class="footnotes-title">${title}</div>\n`

    const separator = options.xhtmlOut
      ? '<hr class="footnotes-sep" />\n'
      : '<hr class="footnotes-sep">\n'

    return `${separator}<section class="footnotes">\n${heading}<ol class="footnotes-list">\n`
  }
}

function safeHeadingAttrs(md) {
  md.core.ruler.before('linkify', 'safe_heading_attrs', (state) => {
    for (let idx = 0; idx < state.tokens.length - 1; idx += 1) {
      const headingOpen = state.tokens[idx]
      const inline = state.tokens[idx + 1]

      if (headingOpen.type !== 'heading_open' || inline.type !== 'inline') {
        continue
      }

      const children = inline.children || []
      const lastText = [...children]
        .reverse()
        .find((token) => token.type === 'text')
      if (!lastText) continue

      const match = lastText.content.match(
        /\s*\{((?:[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)(?:\s+[#.][A-Za-z0-9][A-Za-z0-9_.:-]*)*)\}$/
      )
      if (!match) continue

      const attrs = match[1].trim().split(/\s+/)
      const classes = []

      for (const attr of attrs) {
        if (attr.startsWith('#')) {
          headingOpen.attrSet('id', attr.slice(1))
        } else if (attr.startsWith('.')) {
          classes.push(attr.slice(1))
        }
      }

      if (classes.length) {
        headingOpen.attrJoin('class', classes.join(' '))
      }

      lastText.content = lastText.content.slice(0, match.index)
      inline.content = inline.content.replace(match[0], '')
    }
  })
}

function isExternalAsset(value) {
  return (
    /^(?:[a-z][a-z0-9+.-]*:)?\/\//i.test(value) || value.startsWith('data:')
  )
}

function resolveMarkdownAsset(relativePagePath, src) {
  if (!src || isExternalAsset(src) || src.startsWith('/')) return null

  const hashIndex = src.indexOf('#')
  const queryIndex = src.indexOf('?')
  const suffixIndexCandidates = [hashIndex, queryIndex].filter(
    (idx) => idx >= 0
  )
  const suffixIndex = suffixIndexCandidates.length
    ? Math.min(...suffixIndexCandidates)
    : -1
  const cleanSrc = suffixIndex >= 0 ? src.slice(0, suffixIndex) : src
  const suffix = suffixIndex >= 0 ? src.slice(suffixIndex) : ''
  const pageDir = path.posix.dirname(relativePagePath || '')
  const sourcePath = path.posix
    .normalize(path.posix.join(pageDir, cleanSrc))
    .replace(/^\.\//, '')

  return {
    sourcePath,
    suffix
  }
}

function optimizedImagesPlugin(md) {
  const imageRule = md.renderer.rules.image

  md.renderer.rules.image = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const src = token.attrGet('src')
    const resolved = resolveMarkdownAsset(env.relativePath, src)
    const asset = resolved && assetManifest.assets?.[resolved.sourcePath]

    if (asset?.status === 'optimized' && asset.optimized) {
      token.attrSet('src', `${asset.optimized}${resolved.suffix}`)
      token.attrSet('data-source-src', `/src/${resolved.sourcePath}`)
    }

    if (!token.attrGet('loading')) {
      token.attrSet('loading', 'lazy')
    }

    if (!token.attrGet('decoding')) {
      token.attrSet('decoding', 'async')
    }

    return imageRule(tokens, idx, options, env, self)
  }
}

function mermaidBlockIndex(tokens, idx) {
  let index = 0

  for (let i = 0; i <= idx; i += 1) {
    const info = tokens[i].info?.trim()
    if (info === 'mermaid' || info === 'mmd') {
      index += 1
    }
  }

  return index
}

function buildMermaidManifestMap() {
  return new Map(
    (assetManifest.mermaid || [])
      .filter((block) => block.status === 'optimized' && block.optimized)
      .map((block) => [`${block.page}:${block.index}`, block])
  )
}

const mermaidManifest = buildMermaidManifestMap()

function optimizedMermaidPlugin(md) {
  const fenceRule = md.renderer.rules.fence

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx]
    const info = token.info.trim()

    if (info === 'mermaid' || info === 'mmd') {
      const index = mermaidBlockIndex(tokens, idx)
      const block = mermaidManifest.get(`${env.relativePath}:${index}`)

      if (block?.optimized) {
        const src = md.utils.escapeHtml(block.optimized)
        const source = md.utils.escapeHtml(block.source)
        const page = md.utils.escapeHtml(block.page)

        return `<p class="mermaid-static"><img src="${src}" alt="Mermaid diagram" data-mermaid-viewer="true" data-source-src="/src/${source}" data-source-page="/src/${page}" data-source-index="${block.index}" loading="lazy" decoding="async"></p>\n`
      }
    }

    return fenceRule(tokens, idx, options, env, self)
  }
}

const zhNav = [
  {
    text: '基础课程',
    activeMatch: '^/(?:preface/|chapter(?:0[1-9]|1[0-4])_)',
    items: [
      { text: '预备知识', link: '/preface/introduction' },
      { text: '基础与经典 RL', link: '/chapter01_cartpole/principles' },
      { text: '深度强化学习', link: '/chapter07_dqn/from-q-to-dqn' }
    ]
  },
  {
    text: '进阶专题',
    activeMatch: '^/chapter(?:1[5-9]|2[0-9]|3[0-2])_',
    items: [
      { text: '大模型对齐', link: '/chapter15_rlhf/base-model-to-assistant' },
      { text: 'Agentic 与多模态', link: '/chapter22_agentic/overview' },
      {
        text: '安全与前沿',
        link: '/chapter30_alignment_failures/classical-failures'
      }
    ]
  },
  { text: '附录', link: '/appendix_industrial_training/training-debugging' }
]

const enNav = [
  {
    text: 'Foundations',
    activeMatch: '^/en/(?:preface/|chapter(?:0[1-9]|1[0-4])_)',
    items: [
      { text: 'Preface', link: '/en/preface/introduction' },
      { text: 'Fundamentals', link: '/en/chapter01_cartpole/principles' },
      { text: 'Deep RL', link: '/en/chapter07_dqn/from-q-to-dqn' }
    ]
  },
  {
    text: 'Advanced Topics',
    activeMatch: '^/en/chapter(?:1[5-9]|2[0-9]|3[0-2])_',
    items: [
      {
        text: 'LLM Alignment',
        link: '/en/chapter15_rlhf/base-model-to-assistant'
      },
      { text: 'Agentic RL', link: '/en/chapter22_agentic/overview' },
      {
        text: 'Safety & Frontiers',
        link: '/en/chapter32_selfplay/self-play-outlook/'
      }
    ]
  },
  {
    text: 'Appendices',
    link: '/en/appendix_industrial_training/training-debugging'
  }
]

function courseNav(items) {
  return items.map((item) =>
    item.items ? { component: 'CourseNavMenu', props: item } : item
  )
}

const zhSidebar = {
  '/': [
    {
      text: '序章 · 导论',
      collapsed: false,
      items: [
        {
          text: '强化学习导论',
          link: '/preface/introduction'
        },
        {
          text: '强化学习发展史',
          link: '/preface/brief-history/'
        },
        {
          text: '环境配置',
          link: '/preface/env-setup'
        }
      ]
    },
    {
      text: 'Part I · 基础与经典强化学习',
      collapsed: false,
      items: [
        {
          text: '1. CartPole 入门',
          collapsed: false,
          items: [
            {
              text: '1.1 跑通 CartPole',
              link: '/chapter01_cartpole/principles'
            },
            {
              text: '1.2 CartPole 原理',
              link: '/chapter01_cartpole/metrics'
            },
            {
              text: '1.3 动手：PPO 训练可视化',
              link: '/chapter01_cartpole/training'
            }
          ]
        },
        {
          text: '2. 强化学习问题与基本定义',
          collapsed: false,
          items: [
            {
              text: '2.1 探索与利用',
              link: '/chapter03_mdp/bandit'
            },
            {
              text: '2.2 MDP 与马尔可夫性',
              link: '/chapter03_mdp/mdp'
            },
            {
              text: '2.3 策略、价值与回报',
              link: '/chapter03_mdp/policy-value'
            }
          ]
        },
        {
          text: '3. 价值函数与贝尔曼方程',
          collapsed: false,
          items: [
            {
              text: '3.1 状态价值与贝尔曼期望方程',
              link: '/chapter03_mdp/value-bellman'
            },
            {
              text: '3.2 动作价值与贝尔曼最优方程',
              link: '/chapter03_mdp/value-q'
            },
            {
              text: '3.3 动手：价值迭代与 Q-Learning',
              link: '/chapter03_mdp/value-experiment'
            }
          ]
        },
        {
          text: '4. 经典强化学习方法',
          collapsed: false,
          items: [
            {
              text: '4.1 动态规划、蒙特卡洛与时序差分',
              link: '/chapter03_mdp/dp-mc-td'
            },
            {
              text: '4.2 策略采样与数据来源',
              link: '/chapter03_mdp/algorithm-taxonomy'
            },
            {
              text: '4.3 奖励函数设计',
              link: '/chapter03_mdp/reward-design'
            }
          ]
        }
      ]
    },
    {
      text: 'Part II · 深度强化学习',
      collapsed: false,
      items: [
        {
          text: '5. 深度 Q 网络',
          collapsed: false,
          items: [
            {
              text: '5.1 从 Q-Learning 到 DQN',
              link: '/chapter07_dqn/from-q-to-dqn'
            },
            {
              text: '5.2 DQN 的结构与实现',
              link: '/chapter07_dqn/dqn-components'
            },
            {
              text: '5.3 动手：LunarLander 实战',
              link: '/chapter07_dqn/lunar-lander'
            },
            {
              text: '5.4 DQN 改进方法',
              link: '/chapter07_dqn/dqn-family'
            },
            {
              text: '5.5 动手：视觉游戏项目',
              link: '/chapter07_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '6. 策略梯度方法',
          collapsed: false,
          items: [
            {
              text: '6.1 策略梯度与 REINFORCE',
              link: '/chapter08_policy_gradient/reinforce'
            },
            {
              text: '6.2 策略梯度改进方法',
              link: '/chapter08_policy_gradient/pg-improvements'
            },
            {
              text: '6.3 动手：摇骰子赌博机',
              link: '/chapter08_policy_gradient/dice-game'
            },
            {
              text: '6.4 动手：REINFORCE 控制 CartPole',
              link: '/chapter08_policy_gradient/cartpole'
            },
            {
              text: '6.5 动手：价值基线控制 CartPole',
              link: '/chapter08_policy_gradient/baseline-experiment'
            },
            {
              text: '6.6 动手：带价值基线的策略梯度',
              link: '/chapter08_policy_gradient/cartpole-baseline'
            }
          ]
        },
        {
          text: '7. Actor-Critic 方法',
          collapsed: false,
          items: [
            {
              text: '7.1 优势函数',
              link: '/chapter09_actor_critic/advantage-function'
            },
            {
              text: '7.2 Actor-Critic 同步更新',
              link: '/chapter09_actor_critic/actor-critic'
            },
            {
              text: '7.3 Critic 训练细节',
              link: '/chapter09_actor_critic/critic-training'
            },
            {
              text: '7.4 动手：Pendulum 连续控制',
              link: '/chapter09_actor_critic/pendulum'
            },
            {
              text: '7.5 动手：AlphaGo 复现',
              link: '/chapter09_actor_critic/alphago'
            },
            {
              text: '7.6 动手：BipedalWalker 双足行走',
              link: '/chapter09_actor_critic/bipedalwalker'
            },
            {
              text: '7.7 动手：Actor-Critic 前沿应用',
              link: '/chapter09_actor_critic/ac-frontier'
            }
          ]
        },
        {
          text: '8. PPO：稳定的 Actor-Critic',
          collapsed: false,
          items: [
            {
              text: '8.1 动手：PPO 控制 BipedalWalker',
              link: '/chapter10_ppo/ppo-bipedal-walker'
            },
            {
              text: '8.2 信任域约束与 PPO-Clip',
              link: '/chapter10_ppo/trust-region-clipping'
            },
            {
              text: '8.3 GAE 优势估计',
              link: '/chapter10_ppo/gae-reward-model'
            },
            {
              text: '8.4 动手：PPO 数学推导',
              link: '/chapter10_ppo/ppo-math'
            },
            {
              text: '8.5 动手：PPO 游戏项目',
              link: '/chapter10_ppo/ppo-game-benchmark'
            },
            {
              text: '8.6 动手：长程任务与规划',
              link: '/chapter10_ppo/rl-long-horizon-planning'
            }
          ]
        },
        {
          text: '9. Off-Policy 与 Model-Based RL',
          collapsed: false,
          items: [
            {
              text: '9.1 确定性策略梯度与 DDPG',
              link: '/chapter11_continuous_control/ddpg'
            },
            {
              text: '9.2 TD3 与 SAC',
              link: '/chapter11_continuous_control/td3-sac'
            },
            {
              text: '9.3 基于模型的强化学习',
              link: '/chapter11_continuous_control/model-based'
            },
            {
              text: '9.4 搜索与世界模型',
              link: '/chapter11_continuous_control/search-world-models'
            }
          ]
        }
      ]
    },
    {
      text: 'Part III · 高级强化学习方法（选修）',
      collapsed: false,
      items: [
        {
          text: '10. 离线强化学习（选修）',
          collapsed: false,
          items: [
            {
              text: '10.1 离线数据与分布偏移',
              link: '/chapter12_offline_rl/offline-data-distribution-shift'
            },
            {
              text: '10.2 基于序列建模的离线强化学习',
              link: '/chapter12_offline_rl/sequence-modeling'
            },
            {
              text: '10.3 离线强化学习与偏好数据',
              link: '/chapter12_offline_rl/experiments'
            }
          ]
        },
        {
          text: '11. 模仿学习、逆强化学习与元强化学习（选修）',
          collapsed: false,
          items: [
            {
              text: '11.1 行为克隆与交互式模仿学习',
              link: '/chapter13_imitation_meta_rl/bc-dagger'
            },
            {
              text: '11.2 逆强化学习与 GAIL',
              link: '/chapter13_imitation_meta_rl/irl-gail'
            },
            {
              text: '11.3 元强化学习与上下文适应',
              link: '/chapter13_imitation_meta_rl/meta-rl'
            }
          ]
        },
        {
          text: '12. 探索、多智能体与分层强化学习（选修）',
          collapsed: false,
          items: [
            {
              text: '12.1 内在动机与探索',
              link: '/chapter14_exploration_marl_hierarchical/intrinsic-motivation-exploration'
            },
            {
              text: '12.2 多智能体强化学习',
              link: '/chapter14_exploration_marl_hierarchical/marl'
            },
            {
              text: '12.3 分层强化学习与世界模型',
              link: '/chapter14_exploration_marl_hierarchical/hierarchical'
            }
          ]
        }
      ]
    },
    {
      text: 'Part IV · 大语言模型对齐与后训练',
      collapsed: false,
      items: [
        {
          text: '13. RLHF 训练流水线',
          collapsed: false,
          items: [
            {
              text: '13.1 从基座模型到指令对齐',
              link: '/chapter15_rlhf/base-model-to-assistant'
            },
            {
              text: '13.2 监督微调 SFT',
              link: '/chapter15_rlhf/imitation-learning-pipeline'
            },
            {
              text: '13.3 AI 反馈与安全原则',
              link: '/chapter21_cai_rlvr/hhh-practice'
            },
            {
              text: '13.4 强化学习微调',
              link: '/chapter15_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '13.5 大规模训练工程',
              link: '/chapter15_rlhf/scaling-to-large-models'
            },
            {
              text: '13.6 对齐评测',
              link: '/chapter15_rlhf/evaluation'
            },
            {
              text: '13.7 扩展实战：Reward Hacking 与数据飞轮',
              link: '/chapter15_rlhf/extended-practice'
            },
            {
              text: '13.8 动手：使用 veRL 和 PPO 训练 GSM8K',
              link: '/chapter15_rlhf/verl-ppo-gsm8k'
            },
            {
              text: '13.9 补充阅读：PPO-RLHF 训练循环',
              link: '/chapter15_rlhf/ppo-rlhf-loop'
            }
          ]
        },
        {
          text: '14. 偏好优化与 DPO',
          collapsed: false,
          items: [
            {
              text: '14.1 DPO 目标与推导',
              link: '/chapter17_dpo/dpo-objective-derivation'
            },
            {
              text: '14.2 DPO 训练与评测指标',
              link: '/chapter17_dpo/metrics'
            },
            {
              text: '14.3 DPO 改进方法',
              link: '/chapter17_dpo/dpo-theory-and-family'
            },
            {
              text: '14.4 动手：DPO 对齐实验',
              link: '/chapter17_dpo/dpo-hands-on'
            }
          ]
        },
        {
          text: '15. GRPO 与可验证奖励',
          collapsed: false,
          items: [
            {
              text: '15.1 GRPO 训练机制',
              link: '/chapter18_grpo/grpo-practice-and-mechanism'
            },
            {
              text: '15.2 DeepSeek-R1-Zero 与 DAPO',
              link: '/chapter18_grpo/deepseek-dapo'
            },
            {
              text: '15.3 动手：构建 RLVR 奖励',
              link: '/chapter18_grpo/rlvr'
            },
            {
              text: '15.4 GRPO 改进方法',
              link: '/chapter18_grpo/grpo-family'
            },
            {
              text: '15.5 强化学习环境与验证器',
              link: '/chapter18_grpo/rl-environments'
            },
            {
              text: '15.6 动手：GRPO 训练金融工具调用',
              link: '/chapter18_grpo/financial-tool-calling-grpo'
            },
            {
              text: '15.7 在线策略蒸馏 OPD',
              link: '/chapter18_grpo/on-policy-distillation'
            },
            {
              text: '15.8 动手：使用 veRL 训练代码生成',
              link: '/chapter18_grpo/verl-code-sandbox'
            }
          ]
        },
        {
          text: '16. 推理模型与推理时计算',
          collapsed: false,
          items: [
            {
              text: '16.1 从语言模型到推理模型',
              link: '/chapter19_reasoning/emergence-and-o1'
            },
            {
              text: '16.2 R1-Zero 纯强化学习推理',
              link: '/chapter19_reasoning/r1-zero-pure-rl-reasoning'
            },
            {
              text: '16.3 Test-Time Scaling',
              link: '/chapter19_reasoning/test-time-scaling'
            },
            {
              text: '16.4 Hybrid Thinking 与预算控制',
              link: '/chapter19_reasoning/hybrid-thinking'
            },
            {
              text: '16.5 自适应思考',
              link: '/chapter19_reasoning/adaptive-thinking'
            },
            {
              text: '16.6 推理链的展示与对齐',
              link: '/chapter19_reasoning/cot-visibility-alignment'
            }
          ]
        },
        {
          text: '17. 过程奖励与推理时搜索',
          collapsed: false,
          items: [
            {
              text: '17.1 结果奖励与过程奖励',
              link: '/chapter20_prm_search/outcome-vs-process'
            },
            {
              text: '17.2 判别式 PRM',
              link: '/chapter20_prm_search/discriminative-prm'
            },
            {
              text: '17.3 生成式 PRM',
              link: '/chapter20_prm_search/generative-prm'
            },
            {
              text: '17.4 形式化 Verifier',
              link: '/chapter20_prm_search/formal-prm'
            },
            {
              text: '17.5 推理时搜索',
              link: '/chapter20_prm_search/inference-time-search'
            },
            {
              text: '17.6 并行推理与答案汇总',
              link: '/chapter20_prm_search/parallel-reasoning-and-summary'
            }
          ]
        },
        {
          text: '18. 大模型 RL 工业实践',
          collapsed: false,
          items: [
            {
              text: '18.1 从单机实验到工业训练',
              link: '/chapter16_llm_rl_industrial/single-machine-to-industrial'
            },
            {
              text: '18.2 工业后训练流水线',
              link: '/chapter16_llm_rl_industrial/industrial-post-training'
            },
            {
              text: '18.3 训练稳定性',
              link: '/chapter16_llm_rl_industrial/modern-industrial-practice'
            },
            {
              text: '18.4 分布式 RL 训练',
              link: '/chapter16_llm_rl_industrial/distributed-sync'
            },
            {
              text: '18.5 大规模 RL 数据工程',
              link: '/chapter16_llm_rl_industrial/data-engineering'
            }
          ]
        }
      ]
    },
    {
      text: 'Part V · Agentic RL',
      collapsed: false,
      items: [
        {
          text: '19. Agentic RL 系统',
          collapsed: false,
          items: [
            {
              text: '19.1 Agentic RL 基础',
              link: '/chapter22_agentic/overview'
            },
            {
              text: '19.2 多轮强化学习',
              link: '/chapter22_agentic/formulation'
            },
            {
              text: '19.3 轨迹信用分配',
              link: '/chapter22_agentic/credit-assignment'
            },
            {
              text: '19.4 工具调用与轨迹生成',
              link: '/chapter22_agentic/tool-use-and-trajectory'
            },
            {
              text: '19.5 Search-Augmented RL',
              link: '/chapter22_agentic/tool-use-agents'
            },
            {
              text: '19.6 Code Interpreter RL',
              link: '/chapter22_agentic/industrial-practice'
            },
            {
              text: '19.7 多智能体协作',
              link: '/chapter22_agentic/multi-agent-swarm'
            },
            {
              text: '19.8 动手：使用 rLLM 训练 DeepCoder Agent',
              link: '/chapter22_agentic/rllm-deepcoder-lab'
            },
            {
              text: '19.9 动手：使用 rLLM 训练金融分析 Agent',
              link: '/chapter22_agentic/rllm-finqa-lab'
            },
            {
              text: '19.10 动手：从零实现 Agentic RL 训练系统',
              link: '/chapter22_agentic/build-agentic-training-system'
            }
          ]
        },
        {
          text: '20. 代码智能体强化学习',
          collapsed: false,
          items: [
            {
              text: '20.1 SWE-RL 基础',
              link: '/chapter23_rl_based_swe/swe-bench-and-rlvr'
            },
            {
              text: '20.2 Code World Model 与 DeepSWE',
              link: '/chapter23_rl_based_swe/world-model-and-deep-swe'
            },
            {
              text: '20.3 Self-Play SWE-RL',
              link: '/chapter23_rl_based_swe/self-play-ssr-and-summary'
            }
          ]
        },
        {
          text: '21. Deep Research 与浏览器 Agent',
          collapsed: false,
          items: [
            {
              text: '21.1 浏览器 RL Harness',
              link: '/chapter24_deep_research/browser-rl-harness'
            },
            {
              text: '21.2 评测基准与开源项目',
              link: '/chapter24_deep_research/deep-research-eval'
            }
          ]
        },
        {
          text: '22. Computer Use 与 GUI Agent',
          collapsed: false,
          items: [
            {
              text: '22.1 GUI Agent 训练',
              link: '/chapter25_computer_use/training'
            },
            {
              text: '22.2 Prompt Injection 与指令层级',
              link: '/chapter25_computer_use/safety-swarm'
            }
          ]
        }
      ]
    },
    {
      text: 'Part VI · 多模态强化学习（选修）',
      collapsed: false,
      items: [
        {
          text: '23. 视觉语言模型 RL（选修）',
          collapsed: false,
          items: [
            {
              text: '23.1 视觉奖励与幻觉',
              link: '/chapter26_vlm/vlm-challenges'
            },
            {
              text: '23.2 视觉反思：带着证据回答',
              link: '/chapter26_vlm/qwen3-vl-reflection'
            },
            {
              text: '23.3 动手：多模态 GRPO',
              link: '/chapter26_vlm/vlm-grpo-hands-on'
            },
            {
              text: '23.4 动手：GeoQA 几何推理',
              link: '/chapter26_vlm/easyr1-geoqa'
            }
          ]
        },
        {
          text: '24. 前沿多模态 RL（选修）',
          collapsed: false,
          items: [
            {
              text: '24.1 音频奖励设计',
              link: '/chapter27_audio_rl/reward-design'
            },
            {
              text: '24.2 从音频奖励到实时 Agent',
              link: '/chapter27_audio_rl/future'
            },
            {
              text: '24.3 VLA 模型',
              link: '/chapter28_vla/embodied-intelligence/'
            },
            {
              text: '24.4 图像生成的强化学习对齐',
              link: '/chapter29_visual_generation/visual-generation-dancegrpo'
            },
            {
              text: '24.5 视频的时间一致性',
              link: '/chapter29_visual_generation/video-generation-modern'
            }
          ]
        }
      ]
    },
    {
      text: 'Part VII · 安全、评测与研究前沿（选修）',
      collapsed: false,
      items: [
        {
          text: '25. 奖励黑客与强化学习评测（选修）',
          collapsed: false,
          items: [
            {
              text: '25.1 奖励与任务的背离',
              link: '/chapter30_alignment_failures/classical-failures'
            },
            {
              text: '25.2 RLVR 的假性收益',
              link: '/chapter30_alignment_failures/modern-incidents'
            },
            {
              text: '25.3 潜伏行为与条件切换',
              link: '/chapter30_alignment_failures/sleeper-and-faking'
            },
            {
              text: '25.4 奖励漏洞的防御',
              link: '/chapter30_alignment_failures/scaling-and-defenses'
            },
            {
              text: '25.5 评测协议与可复现性',
              link: '/chapter30_alignment_failures/rl-evaluation'
            }
          ]
        },
        {
          text: '26. 自博弈、规模化与研究前沿（选修）',
          collapsed: false,
          items: [
            {
              text: '26.1 自博弈与训练数据生成',
              link: '/chapter32_selfplay/self-play-outlook/'
            },
            {
              text: '26.2 训练时与测试时的规模扩展',
              link: '/chapter32_selfplay/rl-scaling-outlook'
            },
            {
              text: '26.3 多智能体协同学习',
              link: '/chapter32_selfplay/llm-multi-agent-rl/'
            },
            {
              text: '26.4 LLM 驱动的算法搜索',
              link: '/chapter32_selfplay/alphaevolve/'
            }
          ]
        }
      ]
    },
    {
      text: '附录',
      items: [
        {
          text: 'A. 训练调试与工程',
          collapsed: false,
          items: [
            {
              text: 'A.1 训练跑偏',
              link: '/appendix_industrial_training/training-debugging'
            },
            {
              text: 'A.2 轨迹生成与策略更新',
              link: '/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'A.3 沙箱环境',
              link: '/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'A.4 评估模型改进',
              link: '/appendix_industrial_training/evaluation-badcase'
            }
          ]
        },
        {
          text: 'B. 核心算法实现',
          collapsed: false,
          items: [
            {
              text: 'B.1 SFT 与 KL 散度',
              link: '/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'B.2 PPO 与 GAE',
              link: '/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'B.3 DPO 方法家族',
              link: '/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'B.4 GRPO 与奖励模型',
              link: '/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'B.5 DAPO',
              link: '/appendix_code_cheatsheet/dapo'
            },
            {
              text: 'B.6 Softmax 与交叉熵',
              link: '/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'B.7 采样方法',
              link: '/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'B.8 注意力机制',
              link: '/appendix_code_cheatsheet/attention-mha'
            }
          ]
        },
        {
          text: 'C. 学习资源与参考资料',
          collapsed: false,
          items: [
            {
              text: 'C.1 学习资源与项目路线',
              link: '/appendix_paper_reading/learning-resources'
            },
            {
              text: 'C.2 GPU 小时估算表',
              link: '/appendix_gpu_hours/gpu-hours-estimation'
            },
            {
              text: 'C.3 训练指标词典',
              link: '/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'C.4 工业实践练习',
              link: '/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'D. 数学基础',
          collapsed: false,
          items: [
            {
              text: 'D.1 线性代数',
              collapsed: false,
              items: [
                {
                  text: 'D.1.1 向量与矩阵',
                  link: '/appendix_math/linear-algebra-basics'
                },
                {
                  text: 'D.1.2 贝尔曼方程的矩阵形式',
                  link: '/appendix_math/linear-algebra-bellman'
                },
                {
                  text: 'D.1.3 点积、范数与函数近似',
                  link: '/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: 'D.1.4 收敛性、特征值与信任域',
                  link: '/appendix_math/linear-algebra-advanced'
                },
                {
                  text: 'D.1.5 线性代数公式速查与练习',
                  link: '/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.2 概率、期望与随机估计',
              collapsed: false,
              items: [
                {
                  text: 'D.2.1 概率、条件概率与期望',
                  link: '/appendix_math/probability-basics'
                },
                {
                  text: 'D.2.2 从随机轨迹到状态价值',
                  link: '/appendix_math/probability-value'
                },
                {
                  text: 'D.2.3 蒙特卡洛、增量平均与重要性采样',
                  link: '/appendix_math/probability-sampling'
                },
                {
                  text: 'D.2.4 轨迹概率、Baseline 与 GAE',
                  link: '/appendix_math/probability-trajectory-td'
                },
                {
                  text: 'D.2.5 贝尔曼期望方程与动作价值',
                  link: '/appendix_math/probability-bellman-advanced'
                },
                {
                  text: 'D.2.6 概率统计公式速查与练习',
                  link: '/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.3 微积分与优化',
              collapsed: false,
              items: [
                {
                  text: 'D.3.1 导数、梯度与链式法则',
                  link: '/appendix_math/calculus-basics'
                },
                {
                  text: 'D.3.2 策略梯度与优势函数',
                  link: '/appendix_math/calculus-policy-gradient'
                },
                {
                  text: 'D.3.3 PPO 裁剪、Adam 与更新稳定性',
                  link: '/appendix_math/calculus-ppo'
                },
                {
                  text: 'D.3.4 策略梯度、Taylor 与 GRPO 推导',
                  link: '/appendix_math/calculus-derivations'
                },
                {
                  text: 'D.3.5 强化学习优化公式汇总',
                  link: '/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: 'D.3.6 微积分与优化公式速查与练习',
                  link: '/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.4 信息论与分布距离',
              collapsed: false,
              items: [
                {
                  text: 'D.4.1 自信息、熵与探索',
                  link: '/appendix_math/information-basics'
                },
                {
                  text: 'D.4.2 交叉熵与 KL 散度',
                  link: '/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'D.4.3 PPO、RLHF 与 DPO 中的信息论',
                  link: '/appendix_math/information-rlhf-dpo'
                },
                {
                  text: 'D.4.4 互信息与表征学习',
                  link: '/appendix_math/information-mutual-info'
                },
                {
                  text: 'D.4.5 对齐方法的信息论公式汇总',
                  link: '/appendix_math/information-advanced-formulas'
                },
                {
                  text: 'D.4.6 信息论公式速查与练习',
                  link: '/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

const enSidebar = {
  '/en/': [
    {
      text: 'Preface · Introduction',
      collapsed: false,
      items: [
        {
          text: 'Introduction to RL',
          link: '/en/preface/introduction'
        },
        {
          text: 'Brief History of RL',
          link: '/en/preface/brief-history/'
        },
        {
          text: 'Environment Setup',
          link: '/en/preface/env-setup'
        }
      ]
    },
    {
      text: 'Fundamentals & Classical RL',
      collapsed: false,
      items: [
        {
          text: '1. CartPole',
          collapsed: false,
          items: [
            {
              text: '1.1 Run CartPole',
              link: '/en/chapter01_cartpole/principles'
            },
            {
              text: '1.2 CartPole Principles',
              link: '/en/chapter01_cartpole/metrics'
            },
            {
              text: '1.3 Hands-on: PPO Training Visualization',
              link: '/en/chapter01_cartpole/training'
            }
          ]
        },
        {
          text: '2. RL Problems and Basic Definitions',
          collapsed: false,
          items: [
            {
              text: '2.1 Exploration and Exploitation',
              link: '/en/chapter03_mdp/bandit'
            },
            {
              text: '2.2 Markov Decision Processes',
              link: '/en/chapter03_mdp/mdp'
            },
            {
              text: '2.3 Policy, Value and Return',
              link: '/en/chapter03_mdp/policy-value'
            }
          ]
        },
        {
          text: '3. Value Functions & Bellman Equations',
          collapsed: false,
          items: [
            {
              text: '3.1 State Values and Bellman Expectation',
              link: '/en/chapter03_mdp/value-bellman'
            },
            {
              text: '3.2 Action Values and Bellman Optimality',
              link: '/en/chapter03_mdp/value-q'
            },
            {
              text: '3.3 Hands-on: Value Function Experiments',
              link: '/en/chapter03_mdp/value-experiment'
            }
          ]
        },
        {
          text: '4. Classical RL Methods',
          collapsed: false,
          items: [
            {
              text: '4.1 Dynamic Programming, Monte Carlo and TD',
              link: '/en/chapter03_mdp/dp-mc-td'
            },
            {
              text: '4.2 Policy Sampling and Data Sources',
              link: '/en/chapter03_mdp/algorithm-taxonomy'
            },
            {
              text: '4.3 Reward Design',
              link: '/en/chapter03_mdp/reward-design'
            }
          ]
        }
      ]
    },
    {
      text: 'Deep Reinforcement Learning',
      collapsed: false,
      items: [
        {
          text: '5. Deep Q-Networks',
          collapsed: false,
          items: [
            {
              text: '5.1 From Q-Learning to DQN',
              link: '/en/chapter07_dqn/from-q-to-dqn'
            },
            {
              text: '5.2 DQN Architecture and Implementation',
              link: '/en/chapter07_dqn/dqn-components'
            },
            {
              text: '5.3 Hands-on: LunarLander',
              link: '/en/chapter07_dqn/lunar-lander'
            },
            {
              text: '5.4 DQN Improvements',
              link: '/en/chapter07_dqn/dqn-family'
            },
            {
              text: '5.5 Hands-on: Visual Game Projects',
              link: '/en/chapter07_dqn/visual-game-projects'
            }
          ]
        },
        {
          text: '6. Policy Gradient Methods',
          collapsed: false,
          items: [
            {
              text: '6.1 Policy Gradient and REINFORCE',
              link: '/en/chapter08_policy_gradient/reinforce'
            },
            {
              text: '6.2 Policy Gradient Improvements',
              link: '/en/chapter08_policy_gradient/pg-improvements'
            },
            {
              text: '6.3 Hands-on: Two-Armed Dice-Game Bandit',
              link: '/en/chapter08_policy_gradient/dice-game'
            },
            {
              text: '6.4 Hands-on: REINFORCE on CartPole',
              link: '/en/chapter08_policy_gradient/cartpole'
            },
            {
              text: '6.5 Hands-on: Value Baseline on CartPole',
              link: '/en/chapter08_policy_gradient/baseline-experiment'
            },
            {
              text: '6.6 Hands-on: Policy Gradient with a Value Baseline',
              link: '/en/chapter08_policy_gradient/cartpole-baseline'
            }
          ]
        },
        {
          text: '7. Actor-Critic Methods',
          collapsed: false,
          items: [
            {
              text: '7.1 Advantage Function',
              link: '/en/chapter09_actor_critic/advantage-function'
            },
            {
              text: '7.2 Actor-Critic Updates',
              link: '/en/chapter09_actor_critic/actor-critic'
            },
            {
              text: '7.3 Critic Training Details',
              link: '/en/chapter09_actor_critic/critic-training'
            },
            {
              text: '7.4 Hands-on: Pendulum Continuous Control',
              link: '/en/chapter09_actor_critic/pendulum'
            },
            {
              text: '7.5 Hands-on: Reproducing AlphaGo',
              link: '/en/chapter09_actor_critic/alphago'
            },
            {
              text: '7.6 Hands-on: BipedalWalker',
              link: '/en/chapter09_actor_critic/bipedalwalker'
            },
            {
              text: '7.7 Hands-on: Actor-Critic Frontier Applications',
              link: '/en/chapter09_actor_critic/ac-frontier'
            }
          ]
        },
        {
          text: '8. PPO: Stable Actor-Critic',
          collapsed: false,
          items: [
            {
              text: '8.1 Hands-on: PPO on BipedalWalker',
              link: '/en/chapter10_ppo/ppo-bipedal-walker'
            },
            {
              text: '8.2 Trust Regions and PPO-Clip',
              link: '/en/chapter10_ppo/trust-region-clipping'
            },
            {
              text: '8.3 GAE',
              link: '/en/chapter10_ppo/gae-reward-model'
            },
            {
              text: '8.4 Hands-on: PPO Mathematical Derivation',
              link: '/en/chapter10_ppo/ppo-math'
            },
            {
              text: '8.5 Hands-on: PPO Game Projects',
              link: '/en/chapter10_ppo/ppo-game-benchmark'
            },
            {
              text: '8.6 Hands-on: Long-Horizon Tasks and Planning',
              link: '/en/chapter10_ppo/rl-long-horizon-planning'
            }
          ]
        },
        {
          text: '9. Continuous Control and Model-Based RL',
          collapsed: false,
          items: [
            {
              text: '9.1 Deterministic Policy Gradients and DDPG',
              link: '/en/chapter11_continuous_control/ddpg'
            },
            {
              text: '9.2 TD3 and SAC',
              link: '/en/chapter11_continuous_control/td3-sac'
            },
            {
              text: '9.3 Model-Based RL',
              link: '/en/chapter11_continuous_control/model-based'
            },
            {
              text: '9.4 Search and World Models',
              link: '/en/chapter11_continuous_control/search-world-models'
            }
          ]
        }
      ]
    },
    {
      text: 'Advanced RL Methods (Elective)',
      collapsed: false,
      items: [
        {
          text: '10. Offline Reinforcement Learning (Elective)',
          collapsed: false,
          items: [
            {
              text: '10.1 Offline Data and Distribution Shift',
              link: '/en/chapter12_offline_rl/offline-data-distribution-shift'
            },
            {
              text: '10.2 Sequence Modeling for Offline RL',
              link: '/en/chapter12_offline_rl/sequence-modeling'
            },
            {
              text: '10.3 Offline RL and Preference Data',
              link: '/en/chapter12_offline_rl/experiments'
            }
          ]
        },
        {
          text: '11. Imitation, Inverse and Meta Reinforcement Learning (Elective)',
          collapsed: false,
          items: [
            {
              text: '11.1 Behavioral Cloning and Interactive Imitation',
              link: '/en/chapter13_imitation_meta_rl/bc-dagger'
            },
            {
              text: '11.2 Inverse RL and GAIL',
              link: '/en/chapter13_imitation_meta_rl/irl-gail'
            },
            {
              text: '11.3 Meta-RL and In-Context Adaptation',
              link: '/en/chapter13_imitation_meta_rl/meta-rl'
            }
          ]
        },
        {
          text: '12. Exploration, Multi-Agent and Hierarchical RL (Elective)',
          collapsed: false,
          items: [
            {
              text: '12.1 Intrinsic Motivation and Exploration',
              link: '/en/chapter14_exploration_marl_hierarchical/intrinsic-motivation-exploration'
            },
            {
              text: '12.2 Multi-Agent Reinforcement Learning',
              link: '/en/chapter14_exploration_marl_hierarchical/marl'
            },
            {
              text: '12.3 Hierarchical RL and World Models',
              link: '/en/chapter14_exploration_marl_hierarchical/hierarchical'
            }
          ]
        }
      ]
    },
    {
      text: 'LLM Alignment & Post-Training',
      collapsed: false,
      items: [
        {
          text: '13. RLHF Pipeline',
          collapsed: false,
          items: [
            {
              text: '13.1 Base Model to Instruction Alignment',
              link: '/en/chapter15_rlhf/base-model-to-assistant'
            },
            {
              text: '13.2 Supervised Fine-Tuning',
              link: '/en/chapter15_rlhf/imitation-learning-pipeline'
            },
            {
              text: '13.3 AI Feedback and Safety Principles',
              link: '/en/chapter21_cai_rlvr/hhh-practice'
            },
            {
              text: '13.4 Reinforcement Learning Fine-Tuning',
              link: '/en/chapter15_rlhf/standard-rlhf-pipeline'
            },
            {
              text: '13.5 Large-Scale Training Engineering',
              link: '/en/chapter15_rlhf/scaling-to-large-models'
            },
            {
              text: '13.6 Alignment Evaluation',
              link: '/en/chapter15_rlhf/evaluation'
            },
            {
              text: '13.7 Extended Practice: Reward Hacking and the Data Flywheel',
              link: '/en/chapter15_rlhf/extended-practice'
            },
            {
              text: '13.8 Hands-on: veRL PPO on GSM8K',
              link: '/en/chapter15_rlhf/verl-ppo-gsm8k'
            },
            {
              text: '13.9 Supplement: The PPO-RLHF Training Loop',
              link: '/en/chapter15_rlhf/ppo-rlhf-loop'
            }
          ]
        },
        {
          text: '14. Preference Optimization and DPO',
          collapsed: false,
          items: [
            {
              text: '14.1 DPO Objective and Derivation',
              link: '/en/chapter17_dpo/dpo-objective-derivation'
            },
            {
              text: '14.2 DPO Training and Evaluation Metrics',
              link: '/en/chapter17_dpo/metrics'
            },
            {
              text: '14.3 DPO Improvements',
              link: '/en/chapter17_dpo/dpo-theory-and-family'
            },
            {
              text: '14.4 Hands-on: A DPO Alignment Experiment',
              link: '/en/chapter17_dpo/dpo-hands-on'
            }
          ]
        },
        {
          text: '15. GRPO and Verifiable Rewards',
          collapsed: false,
          items: [
            {
              text: '15.1 GRPO Training Mechanism',
              link: '/en/chapter18_grpo/grpo-practice-and-mechanism'
            },
            {
              text: '15.2 DeepSeek-R1-Zero and DAPO',
              link: '/en/chapter18_grpo/deepseek-dapo'
            },
            {
              text: '15.3 Hands-on: Building RLVR Rewards',
              link: '/en/chapter18_grpo/rlvr'
            },
            {
              text: '15.4 GRPO Improvements',
              link: '/en/chapter18_grpo/grpo-family'
            },
            {
              text: '15.5 RL Environments and Verifiers',
              link: '/en/chapter18_grpo/rl-environments'
            },
            {
              text: '15.6 Hands-on: GRPO for Financial Tool Calling',
              link: '/en/chapter18_grpo/financial-tool-calling-grpo'
            },
            {
              text: '15.7 On-Policy Distillation',
              link: '/en/chapter18_grpo/on-policy-distillation'
            },
            {
              text: '15.8 Hands-on: Code Generation with veRL',
              link: '/en/chapter18_grpo/verl-code-sandbox'
            }
          ]
        },
        {
          text: '16. Reasoning Models and Test-Time Compute',
          collapsed: false,
          items: [
            {
              text: '16.1 From Language Models to Reasoning Models',
              link: '/en/chapter19_reasoning/emergence-and-o1'
            },
            {
              text: '16.2 R1-Zero Pure RL Reasoning',
              link: '/en/chapter19_reasoning/r1-zero-pure-rl-reasoning'
            },
            {
              text: '16.3 Test-Time Scaling',
              link: '/en/chapter19_reasoning/test-time-scaling'
            },
            {
              text: '16.4 Hybrid Thinking and Budget Control',
              link: '/en/chapter19_reasoning/hybrid-thinking'
            },
            {
              text: '16.5 Adaptive Thinking',
              link: '/en/chapter19_reasoning/adaptive-thinking'
            },
            {
              text: '16.6 Reasoning Trace Presentation and Alignment',
              link: '/en/chapter19_reasoning/cot-visibility-alignment'
            }
          ]
        },
        {
          text: '17. Process Rewards and Inference-Time Search',
          collapsed: false,
          items: [
            {
              text: '17.1 Outcome and Process Rewards',
              link: '/en/chapter20_prm_search/outcome-vs-process'
            },
            {
              text: '17.2 Discriminative PRMs',
              link: '/en/chapter20_prm_search/discriminative-prm'
            },
            {
              text: '17.3 Generative PRMs',
              link: '/en/chapter20_prm_search/generative-prm'
            },
            {
              text: '17.4 Formal Verifiers',
              link: '/en/chapter20_prm_search/formal-prm'
            },
            {
              text: '17.5 Inference-Time Search',
              link: '/en/chapter20_prm_search/inference-time-search'
            },
            {
              text: '17.6 Parallel Reasoning and Answer Aggregation',
              link: '/en/chapter20_prm_search/parallel-reasoning-and-summary'
            }
          ]
        },
        {
          text: '18. Industrial LLM RL Engineering',
          collapsed: false,
          items: [
            {
              text: '18.1 From Single-Machine Experiments to Industrial Training',
              link: '/en/chapter16_llm_rl_industrial/single-machine-to-industrial'
            },
            {
              text: '18.2 Industrial Post-Training Pipeline',
              link: '/en/chapter16_llm_rl_industrial/industrial-post-training'
            },
            {
              text: '18.3 Training Stability',
              link: '/en/chapter16_llm_rl_industrial/modern-industrial-practice'
            },
            {
              text: '18.4 Distributed RL Training',
              link: '/en/chapter16_llm_rl_industrial/distributed-sync'
            },
            {
              text: '18.5 Data Engineering for Large-Scale RL',
              link: '/en/chapter16_llm_rl_industrial/data-engineering'
            }
          ]
        }
      ]
    },
    {
      text: 'Agentic RL',
      collapsed: false,
      items: [
        {
          text: '19. Agentic RL Systems',
          collapsed: false,
          items: [
            {
              text: '19.1 Agentic RL Foundations',
              link: '/en/chapter22_agentic/overview'
            },
            {
              text: '19.2 Multi-Turn Reinforcement Learning',
              link: '/en/chapter22_agentic/formulation'
            },
            {
              text: '19.3 Trajectory Credit Assignment',
              link: '/en/chapter22_agentic/credit-assignment'
            },
            {
              text: '19.4 Tool Use and Trajectory Generation',
              link: '/en/chapter22_agentic/tool-use-and-trajectory'
            },
            {
              text: '19.5 Search-Augmented RL',
              link: '/en/chapter22_agentic/tool-use-agents'
            },
            {
              text: '19.6 Code Interpreter RL',
              link: '/en/chapter22_agentic/industrial-practice'
            },
            {
              text: '19.7 Multi-Agent Collaboration',
              link: '/en/chapter22_agentic/multi-agent-swarm'
            },
            {
              text: '19.8 Hands-on: Training a DeepCoder Agent with rLLM',
              link: '/en/chapter22_agentic/rllm-deepcoder-lab'
            },
            {
              text: '19.9 Hands-on: Training a Financial Analysis Agent with rLLM',
              link: '/en/chapter22_agentic/rllm-finqa-lab'
            },
            {
              text: '19.10 Hands-on: Building an Agentic RL Training System from Scratch',
              link: '/en/chapter22_agentic/build-agentic-training-system'
            }
          ]
        },
        {
          text: '20. RL for Code Agents',
          collapsed: false,
          items: [
            {
              text: '20.1 SWE-RL Basics',
              link: '/en/chapter23_rl_based_swe/swe-bench-and-rlvr'
            },
            {
              text: '20.2 Code World Model and DeepSWE',
              link: '/en/chapter23_rl_based_swe/world-model-and-deep-swe'
            },
            {
              text: '20.3 Self-Play SWE-RL',
              link: '/en/chapter23_rl_based_swe/self-play-ssr-and-summary'
            }
          ]
        },
        {
          text: '21. Deep Research and Browser Agents',
          collapsed: false,
          items: [
            {
              text: '21.1 Browser RL Harness',
              link: '/en/chapter24_deep_research/browser-rl-harness'
            },
            {
              text: '21.2 Evaluation Benchmarks and Open-Source Projects',
              link: '/en/chapter24_deep_research/deep-research-eval'
            }
          ]
        },
        {
          text: '22. Computer Use and GUI Agents',
          collapsed: false,
          items: [
            {
              text: '22.1 GUI Agent Training',
              link: '/en/chapter25_computer_use/training'
            },
            {
              text: '22.2 Prompt Injection and Instruction Hierarchy',
              link: '/en/chapter25_computer_use/safety-swarm'
            }
          ]
        }
      ]
    },
    {
      text: 'Multimodal Reinforcement Learning (Elective)',
      collapsed: false,
      items: [
        {
          text: '23. Vision-Language Model RL (Elective)',
          collapsed: false,
          items: [
            {
              text: '23.1 Visual Reward Design',
              link: '/en/chapter26_vlm/vlm-challenges'
            },
            {
              text: '23.2 Visual Reflection RL',
              link: '/en/chapter26_vlm/qwen3-vl-reflection'
            },
            {
              text: '23.3 Hands-on: Multimodal GRPO',
              link: '/en/chapter26_vlm/vlm-grpo-hands-on'
            },
            {
              text: '23.4 Hands-on: GeoQA Geometric Reasoning',
              link: '/en/chapter26_vlm/easyr1-geoqa'
            }
          ]
        },
        {
          text: '24. Frontier Multimodal RL (Elective)',
          collapsed: false,
          items: [
            {
              text: '24.1 Audio Reward Design',
              link: '/en/chapter27_audio_rl/reward-design'
            },
            {
              text: '24.2 Multimodal Audio Agents',
              link: '/en/chapter27_audio_rl/future'
            },
            {
              text: '24.3 VLA Models',
              link: '/en/chapter28_vla/embodied-intelligence/'
            },
            {
              text: '24.4 How RL Aligns Image Generation',
              link: '/en/chapter29_visual_generation/visual-generation-dancegrpo'
            },
            {
              text: '24.5 Why Videos Contradict Themselves',
              link: '/en/chapter29_visual_generation/video-generation-modern'
            }
          ]
        }
      ]
    },
    {
      text: 'Safety, Evaluation and Research Frontiers (Elective)',
      collapsed: false,
      items: [
        {
          text: '25. Reward Hacking and RL Evaluation (Elective)',
          collapsed: false,
          items: [
            {
              text: '25.1 Why Higher Reward Can Make the Task Worse',
              link: '/en/chapter30_alignment_failures/classical-failures'
            },
            {
              text: '25.2 How to Verify RLVR Gains',
              link: '/en/chapter30_alignment_failures/modern-incidents'
            },
            {
              text: '25.3 Why Models Switch Behavior',
              link: '/en/chapter30_alignment_failures/sleeper-and-faking'
            },
            {
              text: '25.4 How to Prevent Reward Exploitation',
              link: '/en/chapter30_alignment_failures/scaling-and-defenses'
            },
            {
              text: '25.5 How to Evaluate RL Models Reliably',
              link: '/en/chapter30_alignment_failures/rl-evaluation'
            }
          ]
        },
        {
          text: '26. Self-Play, Scaling and Research Frontiers (Elective)',
          collapsed: false,
          items: [
            {
              text: '26.1 How Models Generate Training Data',
              link: '/en/chapter32_selfplay/self-play-outlook/'
            },
            {
              text: '26.2 Where Additional Compute Should Go',
              link: '/en/chapter32_selfplay/rl-scaling-outlook'
            },
            {
              text: '26.3 How Multiple LLMs Learn Together',
              link: '/en/chapter32_selfplay/llm-multi-agent-rl/'
            },
            {
              text: '26.4 How LLMs Search for New Algorithms',
              link: '/en/chapter32_selfplay/alphaevolve/'
            }
          ]
        }
      ]
    },
    {
      text: 'Appendices',
      collapsed: false,
      items: [
        {
          text: 'A. Training Debugging and Engineering Practice',
          collapsed: false,
          items: [
            {
              text: 'A.1 Training Debugging Guide',
              link: '/en/appendix_industrial_training/training-debugging'
            },
            {
              text: 'A.2 Training Infrastructure',
              link: '/en/appendix_industrial_training/rl-infrastructure'
            },
            {
              text: 'A.3 Agent Sandbox',
              link: '/en/appendix_industrial_training/agentic-rl-infra'
            },
            {
              text: 'A.4 Evaluation Benchmarks',
              link: '/en/appendix_industrial_training/evaluation-badcase'
            }
          ]
        },
        {
          text: 'B. Core Algorithm Implementations',
          collapsed: false,
          items: [
            {
              text: 'B.1 SFT and KL Divergence',
              link: '/en/appendix_code_cheatsheet/sft-kl'
            },
            {
              text: 'B.2 PPO and GAE',
              link: '/en/appendix_code_cheatsheet/ppo-gae'
            },
            {
              text: 'B.3 DPO Methods',
              link: '/en/appendix_code_cheatsheet/dpo-family'
            },
            {
              text: 'B.4 GRPO and Reward Models',
              link: '/en/appendix_code_cheatsheet/grpo-rlvr'
            },
            {
              text: 'B.5 DAPO',
              link: '/en/appendix_code_cheatsheet/dapo'
            },
            {
              text: 'B.6 Softmax and Cross-Entropy',
              link: '/en/appendix_code_cheatsheet/softmax-ce'
            },
            {
              text: 'B.7 Sampling Methods',
              link: '/en/appendix_code_cheatsheet/top-k-top-p'
            },
            {
              text: 'B.8 Attention Mechanism',
              link: '/en/appendix_code_cheatsheet/attention-mha'
            }
          ]
        },
        {
          text: 'C. Learning Resources and Reference Materials',
          collapsed: false,
          items: [
            {
              text: 'C.1 Learning Resources and Project Roadmap',
              link: '/en/appendix_paper_reading/learning-resources'
            },
            {
              text: 'C.2 GPU Hours Estimation Table',
              link: '/en/appendix_gpu_hours/gpu-hours-estimation'
            },
            {
              text: 'C.3 Metrics Glossary',
              link: '/en/appendix_industrial_training/metrics-glossary'
            },
            {
              text: 'C.4 Industrial Practice Exercises',
              link: '/en/appendix_industrial_training/industrial-exercises'
            }
          ]
        },
        {
          text: 'D. Math Foundations',
          collapsed: false,
          items: [
            {
              text: 'D.1 Linear Algebra',
              collapsed: false,
              items: [
                {
                  text: 'D.1.1 Vectors and Matrices',
                  link: '/en/appendix_math/linear-algebra-basics'
                },
                {
                  text: 'D.1.2 Matrix Form of the Bellman Equation',
                  link: '/en/appendix_math/linear-algebra-bellman'
                },
                {
                  text: 'D.1.3 Dot Products, Norms, and Function Approximation',
                  link: '/en/appendix_math/linear-algebra-function-approx'
                },
                {
                  text: 'D.1.4 Convergence, Eigenvalues, and Trust Regions',
                  link: '/en/appendix_math/linear-algebra-advanced'
                },
                {
                  text: 'D.1.5 Formula Review and Exercises',
                  link: '/en/appendix_math/linear-algebra-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.2 Probability, Expectation, and Random Estimation',
              collapsed: false,
              items: [
                {
                  text: 'D.2.1 Probability, Conditional Probability, and Expectation',
                  link: '/en/appendix_math/probability-basics'
                },
                {
                  text: 'D.2.2 From Random Trajectories to State Values',
                  link: '/en/appendix_math/probability-value'
                },
                {
                  text: 'D.2.3 Monte Carlo, Incremental Averages, and Importance Sampling',
                  link: '/en/appendix_math/probability-sampling'
                },
                {
                  text: 'D.2.4 Trajectory Probability, Baselines, and GAE',
                  link: '/en/appendix_math/probability-trajectory-td'
                },
                {
                  text: 'D.2.5 Bellman Expectation Equation and Action Values',
                  link: '/en/appendix_math/probability-bellman-advanced'
                },
                {
                  text: 'D.2.6 Probability and Statistics Formula Reference and Exercises',
                  link: '/en/appendix_math/probability-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.3 Calculus and Optimization',
              collapsed: false,
              items: [
                {
                  text: 'D.3.1 Derivatives, Gradients, and the Chain Rule',
                  link: '/en/appendix_math/calculus-basics'
                },
                {
                  text: 'D.3.2 Policy Gradients and Advantage Functions',
                  link: '/en/appendix_math/calculus-policy-gradient'
                },
                {
                  text: 'D.3.3 PPO Clipping, Adam, and Update Stability',
                  link: '/en/appendix_math/calculus-ppo'
                },
                {
                  text: 'D.3.4 Policy Gradient, Taylor, and GRPO Derivations',
                  link: '/en/appendix_math/calculus-derivations'
                },
                {
                  text: 'D.3.5 Complete Formulas for PG, DQN, GAE, PPO, and GRPO',
                  link: '/en/appendix_math/calculus-advanced-formulas'
                },
                {
                  text: 'D.3.6 Calculus and Optimization Formula Reference and Exercises',
                  link: '/en/appendix_math/calculus-formulas-exercises'
                }
              ]
            },
            {
              text: 'D.4 Information Theory and Distribution Distance',
              collapsed: false,
              items: [
                {
                  text: 'D.4.1 Self-Information, Entropy, and Exploration',
                  link: '/en/appendix_math/information-basics'
                },
                {
                  text: 'D.4.2 Cross-Entropy and KL Divergence',
                  link: '/en/appendix_math/information-cross-entropy-kl'
                },
                {
                  text: 'D.4.3 Information Theory in PPO, RLHF, and DPO',
                  link: '/en/appendix_math/information-rlhf-dpo'
                },
                {
                  text: 'D.4.4 Mutual Information and Representation Learning',
                  link: '/en/appendix_math/information-mutual-info'
                },
                {
                  text: 'D.4.5 Complete Formulas for KL, RLHF, DPO, and Mutual Information',
                  link: '/en/appendix_math/information-advanced-formulas'
                },
                {
                  text: 'D.4.6 Information Theory Formula Reference and Exercises',
                  link: '/en/appendix_math/information-formulas-exercises'
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

function collectEnglishRoutes(directory, relativeDirectory = '') {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const relativePath = path.posix.join(relativeDirectory, entry.name)
    const absolutePath = path.join(directory, entry.name)

    if (entry.isDirectory()) {
      return collectEnglishRoutes(absolutePath, relativePath)
    }

    if (!entry.isFile() || !entry.name.endsWith('.md')) return []

    const routePath = relativePath.replace(/\.md$/, '').replace(/\/index$/, '/')
    return [routePath === 'index' ? '/en/' : `/en/${routePath}`]
  })
}

const englishRoutes = collectEnglishRoutes(path.join(docsRoot, 'en'))
const chineseRoutes = englishRoutes
  .map((route) => route.replace(/^\/en(?=\/|$)/, '') || '/')
  .filter(
    (route) =>
      fs.existsSync(path.join(docsRoot, `${route}.md`)) ||
      fs.existsSync(path.join(docsRoot, route, 'index.md'))
  )

const logger = createLogger()
const originalWarn = logger.warn
logger.warn = (msg, options) => {
  if (msg.includes('Failed to resolve "/@siteData"')) return
  originalWarn(msg, options)
}

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hands-on Modern RL',
  description:
    '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体',
  base,
  cleanUrls: true,
  lastUpdated: true,
  markdown: {
    image: {
      lazyLoading: true
    },
    attrs: {
      disable: true
    },
    config: (md) => {
      safeHeadingAttrs(md)
      optimizedImagesPlugin(md)
      md.use(markdownItFootnote)
      footnoteTitlePlugin(md)
      katexMarkdown(md)
      MermaidMarkdown(md)
      optimizedMermaidPlugin(md)
      // Custom "output" container for displaying code running results
      md.use(markdownItContainer, 'output', {
        render: function (tokens, idx) {
          if (tokens[idx].nesting === 1) {
            const title = tokens[idx].info.trim().slice(6).trim() || '运行结果'
            return `<div class="custom-block output"><p class="custom-block-title">${title}</p>\n`
          }
          return '</div>\n'
        }
      })
    }
  },
  vite: {
    customLogger: logger,
    server: {
      watch: {
        ignored: ['**/.vitepress/dist/**']
      }
    },
    plugins: [mermaidConfigPlugin(), normalizeBrokenDocPathPlugin()],
    optimizeDeps: {
      include: [
        '@braintree/sanitize-url',
        'cytoscape',
        'cytoscape-cose-bilkent',
        'dayjs',
        'debug'
      ]
    },
    resolve: {
      alias: {
        'dayjs/plugin/advancedFormat.js': 'dayjs/esm/plugin/advancedFormat',
        'dayjs/plugin/customParseFormat.js':
          'dayjs/esm/plugin/customParseFormat',
        'dayjs/plugin/isoWeek.js': 'dayjs/esm/plugin/isoWeek',
        'cytoscape/dist/cytoscape.umd.js': 'cytoscape/dist/cytoscape.esm.js'
      }
    }
  },
  ignoreDeadLinks: false,
  head: [
    ['link', { rel: 'icon', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#3f51b5' }],
    [
      'meta',
      { name: 'viewport', content: 'width=device-width, initial-scale=1.0' }
    ],
    ['meta', { name: 'author', content: 'WalkingLabs' }],
    ['meta', { name: 'robots', content: 'index,follow' }],
    ['meta', { property: 'og:title', content: 'Hands-on Modern RL' }],
    [
      'meta',
      {
        property: 'og:description',
        content:
          '现代强化学习实战指南：涵盖经典控制、LLM 后训练、RLVR 与多模态智能体'
      }
    ],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:url', content: siteUrl }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }]
  ],
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/',
      title: 'Hands-on Modern RL',
      description: '现代强化学习实战——从代码到原理',
      themeConfig: {
        englishRoutes,
        chineseRoutes,
        nav: courseNav(zhNav),
        sidebar: zhSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: '在 GitHub 上编辑此页'
        },
        footer: {
          message: '现代强化学习实战课程',
          copyright: 'Copyright © WalkingLabs'
        },
        outline: {
          level: [2, 3],
          label: '大纲'
        },
        lastUpdated: {
          text: '最后更新'
        },
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        returnToTopLabel: '返回顶部',
        langMenuLabel: '切换语言',
        skipToContentLabel: '跳转到正文',
        notFound: {
          title: '页面未找到',
          quote: '这个地址不存在，试试从中文首页重新进入。',
          link: '/',
          linkText: '返回中文首页',
          linkLabel: '返回中文首页'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Hands-on Modern RL',
      description:
        'Modern Reinforcement Learning in Practice — From Code to Theory',
      themeConfig: {
        englishRoutes,
        chineseRoutes,
        nav: courseNav(enNav),
        sidebar: enSidebar,
        editLink: {
          pattern: editLinkPattern,
          text: 'Edit this page on GitHub'
        },
        footer: {
          message: 'Hands-on Modern Reinforcement Learning',
          copyright: 'Copyright © WalkingLabs'
        },
        outline: {
          level: [2, 3],
          label: 'On this page'
        },
        lastUpdated: {
          text: 'Last updated'
        },
        docFooter: {
          prev: 'Previous page',
          next: 'Next page'
        },
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        returnToTopLabel: 'Return to top',
        langMenuLabel: 'Change language',
        skipToContentLabel: 'Skip to content',
        notFound: {
          title: 'Page not found',
          quote:
            'This page is missing. Try jumping back in from the English home page.',
          link: '/en/',
          linkText: 'Take me to English home',
          linkLabel: 'Go to English home'
        }
      }
    }
  },
  themeConfig: {
    // The custom language toggle resolves matching pages; stock menus use home.
    i18nRouting: false,
    logo: '/readme/logo-symbol.svg',
    siteTitle: 'Hands on Modern RL',
    nav: courseNav(zhNav),
    sidebar: zhSidebar,
    socialLinks: [
      { icon: 'github', link: `https://github.com/${owner}/${repo}` }
    ],
    search: enableLocalSearch
      ? {
          provider: 'local',
          options: {
            _render: renderSearchMarkdown
          }
        }
      : undefined,
    editLink: {
      pattern: editLinkPattern,
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Built for reusable bilingual course delivery',
      copyright: 'Copyright © WalkingLabs'
    },
    outline: {
      level: [2, 3],
      label: '大纲'
    }
  }
})
