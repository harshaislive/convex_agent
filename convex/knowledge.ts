import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

const DEFAULT_MAX_RESULTS = 5;

function normalizeTagList(values: string[] | undefined): string[] {
  if (!values) return [];
  const seen = new Set<string>();
  const normalized: string[] = [];
  for (const value of values) {
    const cleaned = value.trim().toLowerCase();
    if (!cleaned || seen.has(cleaned)) continue;
    seen.add(cleaned);
    normalized.push(cleaned);
  }
  return normalized;
}

function slugify(value: string): string {
  const slug = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || "entry";
}

function buildSearchText(args: {
  title: string;
  summary?: string;
  body: string;
  type: string;
  tags: string[];
  intentTags: string[];
  audienceTags: string[];
}): string {
  return [
    args.title,
    args.summary || "",
    args.body,
    args.type,
    args.tags.join(" "),
    args.intentTags.join(" "),
    args.audienceTags.join(" "),
  ]
    .join("\n")
    .toLowerCase();
}

function scoreEntry(
  queryText: string,
  entry: {
    searchText: string;
    priority: number;
    updatedAt: number;
    intentTags: string[];
    audienceTags: string[];
    tags: string[];
  },
  intent: string | undefined,
  audience: string | undefined,
): number {
  const terms = queryText
    .toLowerCase()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean);

  let score = 0;
  for (const term of terms) {
    const count = entry.searchText.split(term).length - 1;
    score += count * 4;
  }

  const normalizedIntent = intent?.trim().toLowerCase();
  if (normalizedIntent && entry.intentTags.includes(normalizedIntent)) {
    score += 18;
  }

  const normalizedAudience = audience?.trim().toLowerCase();
  if (normalizedAudience && entry.audienceTags.includes(normalizedAudience)) {
    score += 12;
  }

  score += entry.priority * 10;

  const ageDays = Math.max(0, (Date.now() - entry.updatedAt) / 86_400_000);
  if (ageDays <= 30) score += 6;
  else if (ageDays <= 90) score += 3;

  return score;
}

export const upsertEntry = mutation({
  args: {
    slug: v.optional(v.string()),
    title: v.string(),
    type: v.string(),
    summary: v.optional(v.string()),
    body: v.string(),
    intentTags: v.optional(v.array(v.string())),
    audienceTags: v.optional(v.array(v.string())),
    tags: v.optional(v.array(v.string())),
    priority: v.optional(v.float64()),
    status: v.optional(v.string()),
    effectiveFrom: v.optional(v.float64()),
    effectiveTo: v.optional(v.float64()),
    sourceType: v.optional(v.string()),
    sourceUrl: v.optional(v.string()),
    owner: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const now = Date.now();
    const slugBase = args.slug?.trim() || slugify(args.title);
    const slug = slugBase || slugify(args.title);
    const intentTags = normalizeTagList(args.intentTags);
    const audienceTags = normalizeTagList(args.audienceTags);
    const tags = normalizeTagList(args.tags);
    const searchText = buildSearchText({
      title: args.title,
      summary: args.summary,
      body: args.body,
      type: args.type,
      tags,
      intentTags,
      audienceTags,
    });

    const existing = await ctx.db
      .query("knowledge_entries")
      .withIndex("by_slug", (q) => q.eq("slug", slug))
      .first();

    const payload = {
      slug,
      title: args.title.trim(),
      type: args.type.trim().toLowerCase(),
      summary: args.summary?.trim() || undefined,
      body: args.body.trim(),
      intentTags,
      audienceTags,
      tags,
      priority: args.priority ?? 0,
      status: (args.status || "approved").trim().toLowerCase(),
      effectiveFrom: args.effectiveFrom,
      effectiveTo: args.effectiveTo,
      sourceType: args.sourceType?.trim().toLowerCase(),
      sourceUrl: args.sourceUrl?.trim() || undefined,
      owner: args.owner?.trim() || undefined,
      searchText,
      updatedAt: now,
    };

    if (existing) {
      await ctx.db.patch(existing._id, payload);
      return { id: existing._id, slug, updated: true };
    }

    const id = await ctx.db.insert("knowledge_entries", {
      ...payload,
      createdAt: now,
    });
    return { id, slug, updated: false };
  },
});

export const listEntries = query({
  args: {
    status: v.optional(v.string()),
    type: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    let entries;
    if (args.status) {
      entries = await ctx.db
        .query("knowledge_entries")
        .withIndex("by_status", (q) => q.eq("status", args.status!.trim().toLowerCase()))
        .collect();
    } else if (args.type) {
      entries = await ctx.db
        .query("knowledge_entries")
        .withIndex("by_type", (q) => q.eq("type", args.type!.trim().toLowerCase()))
        .collect();
    } else {
      entries = await ctx.db.query("knowledge_entries").collect();
    }

    return entries
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .map((entry) => ({
        slug: entry.slug,
        title: entry.title,
        type: entry.type,
        summary: entry.summary,
        tags: entry.tags,
        intentTags: entry.intentTags,
        audienceTags: entry.audienceTags,
        priority: entry.priority,
        status: entry.status,
        sourceType: entry.sourceType,
        sourceUrl: entry.sourceUrl,
        updatedAt: entry.updatedAt,
      }));
  },
});

export const getEntry = query({
  args: {
    slug: v.string(),
  },
  handler: async (ctx, args) => {
    const entry = await ctx.db
      .query("knowledge_entries")
      .withIndex("by_slug", (q) => q.eq("slug", args.slug))
      .first();

    if (!entry) {
      return null;
    }

    return {
      slug: entry.slug,
      title: entry.title,
      type: entry.type,
      summary: entry.summary,
      body: entry.body,
      tags: entry.tags,
      intentTags: entry.intentTags,
      audienceTags: entry.audienceTags,
      priority: entry.priority,
      status: entry.status,
      sourceType: entry.sourceType,
      sourceUrl: entry.sourceUrl,
      updatedAt: entry.updatedAt,
    };
  },
});

export const deleteEntry = mutation({
  args: {
    slug: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("knowledge_entries")
      .withIndex("by_slug", (q) => q.eq("slug", args.slug))
      .first();

    if (!existing) {
      return { deleted: false };
    }

    await ctx.db.delete(existing._id);
    return { deleted: true, slug: args.slug };
  },
});

export const searchEntries = query({
  args: {
    query: v.string(),
    maxResults: v.optional(v.float64()),
    intent: v.optional(v.string()),
    audience: v.optional(v.string()),
    status: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const requestedStatus = (args.status || "approved").trim().toLowerCase();
    const maxResults = Math.max(1, Math.min(20, Math.floor(args.maxResults || DEFAULT_MAX_RESULTS)));
    const now = Date.now();
    const rawQuery = args.query.trim();
    if (!rawQuery) return [];

    const entries = await ctx.db
      .query("knowledge_entries")
      .withIndex("by_status", (q) => q.eq("status", requestedStatus))
      .collect();

    const ranked = entries
      .filter((entry) => {
        if (entry.effectiveFrom && entry.effectiveFrom > now) return false;
        if (entry.effectiveTo && entry.effectiveTo < now) return false;
        return true;
      })
      .map((entry) => ({
        ...entry,
        score: scoreEntry(rawQuery, entry, args.intent, args.audience),
      }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return b.updatedAt - a.updatedAt;
      })
      .slice(0, maxResults);

    return ranked.map((entry) => ({
      slug: entry.slug,
      title: entry.title,
      type: entry.type,
      summary: entry.summary,
      body: entry.body,
      tags: entry.tags,
      intentTags: entry.intentTags,
      audienceTags: entry.audienceTags,
      priority: entry.priority,
      status: entry.status,
      sourceType: entry.sourceType,
      sourceUrl: entry.sourceUrl,
      updatedAt: entry.updatedAt,
      score: entry.score,
    }));
  },
});
