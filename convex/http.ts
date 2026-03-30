import { httpRouter } from "convex/server";
import { httpAction } from "./_generated/server";
import { api } from "./_generated/api";

const http = httpRouter();

function omitNullishFields(value: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(value).filter(([, fieldValue]) => fieldValue !== null && fieldValue !== undefined),
  );
}

http.route({
  path: "/instagram/store-dm-event",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const expectedSecret = process.env.AGENT_SHARED_SECRET;
    const providedSecret = req.headers.get("x-agent-secret");

    if (!expectedSecret || providedSecret !== expectedSecret) {
      return new Response("Unauthorized", { status: 401 });
    }

    const body = await req.json();
    await ctx.runMutation(api.instagramDm.storeAgentDmEvent, body);
    return Response.json({ ok: true });
  }),
});

http.route({
  path: "/instagram/conversation-history",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const providedSecret = req.headers.get("x-agent-secret");
    const deploymentSecret = process.env.AGENT_SHARED_SECRET;
    if (!deploymentSecret || providedSecret !== deploymentSecret) {
      return new Response("Unauthorized", { status: 401 });
    }
    const url = new URL(req.url);
    const contactId = url.searchParams.get("contactId");
    if (!contactId) {
      return new Response("contactId is required", { status: 400 });
    }
    const messages = await ctx.runQuery(api.instagramDm.getMessagesByContactId, {
      contactId,
    });
    return Response.json(messages);
  }),
});

http.route({
  path: "/instagram/append-message",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const providedSecret = req.headers.get("x-agent-secret");
    const deploymentSecret = process.env.AGENT_SHARED_SECRET;
    if (!deploymentSecret || providedSecret !== deploymentSecret) {
      return new Response("Unauthorized", { status: 401 });
    }
    const body = await req.json();
    await ctx.runMutation(api.signalConversations.appendInstagramStyleMessage, {
      contactId: body.contactId,
      role: body.role,
      content: body.content,
      timestamp: body.timestamp,
    });
    return Response.json({ ok: true });
  }),
});

http.route({
  path: "/knowledge/search",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const providedSecret = req.headers.get("x-agent-secret");
    const deploymentSecret = process.env.AGENT_SHARED_SECRET;
    if (!deploymentSecret || providedSecret !== deploymentSecret) {
      return new Response("Unauthorized", { status: 401 });
    }

    const url = new URL(req.url);
    const query = url.searchParams.get("query") || "";
    const intent = url.searchParams.get("intent") || undefined;
    const audience = url.searchParams.get("audience") || undefined;
    const maxResults = Number(url.searchParams.get("maxResults") || "5");

    if (!query.trim()) {
      return new Response("query is required", { status: 400 });
    }

    const results = await ctx.runQuery(api.knowledge.searchEntries, {
      query,
      intent,
      audience,
      maxResults,
    });
    return Response.json(results);
  }),
});

http.route({
  path: "/knowledge/entries",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const providedSecret = req.headers.get("x-agent-secret");
    const deploymentSecret = process.env.AGENT_SHARED_SECRET;
    if (!deploymentSecret || providedSecret !== deploymentSecret) {
      return new Response("Unauthorized", { status: 401 });
    }

    const url = new URL(req.url);
    const status = url.searchParams.get("status") || undefined;
    const type = url.searchParams.get("type") || undefined;
    const entries = await ctx.runQuery(api.knowledge.listEntries, {
      status,
      type,
    });
    return Response.json(entries);
  }),
});

http.route({
  path: "/knowledge/entry",
  method: "GET",
  handler: httpAction(async (ctx, req) => {
    const providedSecret = req.headers.get("x-agent-secret");
    const deploymentSecret = process.env.AGENT_SHARED_SECRET;
    if (!deploymentSecret || providedSecret !== deploymentSecret) {
      return new Response("Unauthorized", { status: 401 });
    }

    const url = new URL(req.url);
    const slug = url.searchParams.get("slug") || "";
    if (!slug.trim()) {
      return new Response("slug is required", { status: 400 });
    }

    const entry = await ctx.runQuery(api.knowledge.getEntry, { slug });
    if (!entry) {
      return new Response("Not found", { status: 404 });
    }
    return Response.json(entry);
  }),
});

http.route({
  path: "/knowledge/upsert-entry",
  method: "POST",
  handler: httpAction(async (ctx, req) => {
    const expectedSecret = process.env.AGENT_SHARED_SECRET;
    const providedSecret = req.headers.get("x-agent-secret");

    if (!expectedSecret || providedSecret !== expectedSecret) {
      return new Response("Unauthorized", { status: 401 });
    }

    const body = await req.json();
    const result = await ctx.runMutation(api.knowledge.upsertEntry, omitNullishFields(body as Record<string, unknown>));
    return Response.json({ ok: true, ...result });
  }),
});

export default http;
