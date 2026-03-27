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

export default http;
