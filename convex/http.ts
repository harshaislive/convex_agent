import { httpRouter } from "convex/server";
import { httpAction } from "./_generated/server";
import { api } from "./_generated/api";

const http = httpRouter();

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

export default http;
