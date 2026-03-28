import { query } from "./_generated/server";
import { v } from "convex/values";

export const getConversationHistory = query({
  args: {
    threadId: v.string(),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    const limit = args.limit ?? 20;
    const messages = await ctx.db
      .query("instagramConversations")
      .withIndex("by_instagram_user_id", (q) => q.eq("instagramUserId", args.threadId))
      .order("asc")
      .take(limit);

    return messages.map((msg) => ({
      message: msg.message,
      agentReplyText: msg.agentReplyText,
      receivedAt: msg.receivedAt,
      agentReplyAt: msg.agentReplyAt,
      name: msg.name,
      instagramAccountName: msg.instagramAccountName,
    }));
  },
});
