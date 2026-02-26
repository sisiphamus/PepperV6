// Memory manager — reads/writes all 4 memory categories from pepperv1/backend/bot/memory/.
// Categories: skills, knowledge, preferences, sites.

import { readdirSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MEMORY_ROOT = join(__dirname, '..', '..', 'pepperv1', 'backend', 'bot', 'memory');

let inventoryCache = null;

function readFirstLine(filepath) {
  try {
    const content = readFileSync(filepath, 'utf-8');
    // Try YAML frontmatter description
    const descMatch = content.match(/^---[\s\S]*?description:\s*(.+)/m);
    if (descMatch) return descMatch[1].trim();
    // Try first heading
    const headingMatch = content.match(/^#\s+(.+)/m);
    if (headingMatch) return headingMatch[1].trim();
    // First non-empty line
    const firstLine = content.split('\n').find(l => l.trim());
    return firstLine ? firstLine.trim().slice(0, 100) : '(no description)';
  } catch {
    return '(unreadable)';
  }
}

function scanCategory(category, subdir) {
  const dir = join(MEMORY_ROOT, subdir);
  if (!existsSync(dir)) return [];

  const results = [];
  const entries = readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    if (entry.isDirectory()) {
      // Skills are in subdirectories with SKILL.md
      const skillFile = join(dir, entry.name, 'SKILL.md');
      if (existsSync(skillFile)) {
        results.push({
          name: entry.name,
          category,
          description: readFirstLine(skillFile),
          path: skillFile,
        });
      }
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      results.push({
        name: entry.name.replace('.md', ''),
        category,
        description: readFirstLine(join(dir, entry.name)),
        path: join(dir, entry.name),
      });
    }
  }

  return results;
}

export function getFullInventory() {
  if (inventoryCache) return inventoryCache;

  const inventory = [
    ...scanCategory('skill', 'skills'),
    ...scanCategory('knowledge', 'knowledge'),
    ...scanCategory('preference', 'preferences'),
    ...scanCategory('site', 'sites'),
  ];

  inventoryCache = inventory;
  return inventory;
}

export function invalidateCache() {
  inventoryCache = null;
}

export function getContents(selections) {
  return selections.map(sel => {
    const inventory = getFullInventory();
    const match = inventory.find(m => m.name === sel.name && m.category === sel.category);
    if (!match) return { ...sel, content: '(not found)' };

    try {
      const content = readFileSync(match.path, 'utf-8');
      return { ...sel, content, path: match.path };
    } catch {
      return { ...sel, content: '(unreadable)' };
    }
  });
}

export function writeMemory(name, category, content) {
  const categoryDirs = {
    skill: 'skills',
    knowledge: 'knowledge',
    preference: 'preferences',
    site: 'sites',
  };

  const subdir = categoryDirs[category];
  if (!subdir) throw new Error(`Unknown memory category: ${category}`);

  let filepath;
  if (category === 'skill') {
    const dir = join(MEMORY_ROOT, subdir, name);
    mkdirSync(dir, { recursive: true });
    filepath = join(dir, 'SKILL.md');
  } else {
    const dir = join(MEMORY_ROOT, subdir);
    mkdirSync(dir, { recursive: true });
    filepath = join(dir, `${name}.md`);
  }

  writeFileSync(filepath, content, 'utf-8');
  invalidateCache();
  return filepath;
}

export function updateMemory(path, action, content) {
  if (action === 'append') {
    const existing = existsSync(path) ? readFileSync(path, 'utf-8') : '';
    writeFileSync(path, existing + '\n\n' + content, 'utf-8');
  } else {
    writeFileSync(path, content, 'utf-8');
  }
  invalidateCache();
}

export function detectSiteContext(prompt) {
  const sitesDir = join(MEMORY_ROOT, 'sites');
  if (!existsSync(sitesDir)) return [];

  const promptLower = prompt.toLowerCase();
  const matches = [];

  try {
    const files = readdirSync(sitesDir).filter(f => f.endsWith('.md'));
    for (const file of files) {
      const name = file.replace('.md', '');
      if (promptLower.includes(name.toLowerCase())) {
        try {
          matches.push({
            name,
            category: 'site',
            content: readFileSync(join(sitesDir, file), 'utf-8'),
          });
        } catch {}
      }
    }
  } catch {}

  return matches;
}
