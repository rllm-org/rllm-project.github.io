#!/usr/bin/env node
// Generates rss.xml from the post registry in posts.js.
// Run after editing posts.js:  node generate_rss.js

const fs = require("fs");
const path = require("path");
const { POSTS } = require("./posts.js");

const SITE_URL = "https://rllm-project.com";
const FEED_TITLE = "rLLM Project — Blog";
const FEED_DESCRIPTION =
  "Research, releases, and updates from the rLLM Project — building the infrastructure to train, evaluate, and evolve intelligent agents.";
const FEED_LANGUAGE = "en-us";

function escapeXml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function absoluteLink(post) {
  if (post.external) return post.link;
  return `${SITE_URL}/${post.link.replace(/^\//, "")}`;
}

function pubDate(isoDate) {
  // Treat the date as UTC midnight for a stable RFC-822 timestamp.
  const [y, m, d] = isoDate.split("-").map((n) => parseInt(n, 10));
  return new Date(Date.UTC(y, m - 1, d)).toUTCString();
}

function buildItem(post) {
  const link = absoluteLink(post);
  return `    <item>
      <title>${escapeXml(post.title)}</title>
      <link>${escapeXml(link)}</link>
      <guid isPermaLink="true">${escapeXml(link)}</guid>
      <pubDate>${pubDate(post.date)}</pubDate>
      <category>${escapeXml(post.category)}</category>
      <dc:creator>${escapeXml(post.authors)}</dc:creator>
      <description>${escapeXml(post.excerpt)}</description>
    </item>`;
}

function buildFeed(posts) {
  const sorted = posts
    .slice()
    .sort((a, b) => new Date(b.date) - new Date(a.date));
  const lastBuild = new Date().toUTCString();
  const latest = sorted.length ? pubDate(sorted[0].date) : lastBuild;
  const items = sorted.map(buildItem).join("\n");

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>${escapeXml(FEED_TITLE)}</title>
    <link>${SITE_URL}/blog.html</link>
    <atom:link href="${SITE_URL}/rss.xml" rel="self" type="application/rss+xml" />
    <description>${escapeXml(FEED_DESCRIPTION)}</description>
    <language>${FEED_LANGUAGE}</language>
    <lastBuildDate>${lastBuild}</lastBuildDate>
    <pubDate>${latest}</pubDate>
${items}
  </channel>
</rss>
`;
}

const outPath = path.join(__dirname, "rss.xml");
fs.writeFileSync(outPath, buildFeed(POSTS), "utf8");
console.log(`Wrote ${outPath} with ${POSTS.length} items.`);
