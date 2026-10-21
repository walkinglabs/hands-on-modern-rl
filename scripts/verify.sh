#!/bin/sh
set -eu

echo "Checking Notebook catalog..."
npm run verify:notebooks

echo "Checking formatting..."
npm run format:check

echo "Checking lint..."
npm run lint

echo "Checking bilingual coverage..."
npm run verify:bilingual

echo "Checking English publication quality..."
npm run verify:english

echo "Building site..."
npm run build

echo "Checking build output..."
test -f docs/.vitepress/dist/preface/intro.html
test -f docs/.vitepress/dist/sitemap.xml
test -f docs/.vitepress/dist/robots.txt

echo "Verification passed"
