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
    const threadId = url.searchParams.get("threadId");
    if (!threadId) {
      return new Response("threadId is required", { status: 400 });
    }
    const limit = parseInt(url.searchParams.get("limit") ?? "20", 10);
    const messages = await ctx.runQuery(api.conversationHistory.getConversationHistory, {
      threadId,
      limit,
    });
    return Response.json(messages);
  }),
});

export default http;
