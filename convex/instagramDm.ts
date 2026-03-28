import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const storeAgentDmEvent = mutation({
  args: {
    contactId: v.string(),
    message: v.optional(v.string()),
    name: v.optional(v.string()),
    instagramUserId: v.optional(v.string()),
    instagramAccountName: v.optional(v.string()),
    igFollowersCount: v.optional(v.float64()),
    igMessagingWindow: v.optional(v.string()),
    isIgAccountFollowUser: v.optional(v.boolean()),
    isIgAccountFollower: v.optional(v.boolean()),
    isIgVerifiedUser: v.optional(v.boolean()),
    lastIgInteraction: v.optional(v.string()),
    lastIgSeen: v.optional(v.string()),
    optinInstagram: v.optional(v.boolean()),
    rawPayload: v.optional(v.any()),
    receivedAt: v.float64(),
    agentReplied: v.boolean(),
    agentReplyAt: v.optional(v.float64()),
    agentReplyText: v.optional(v.string()),
    lastReplyType: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("instagramConversations", args);
  },
});

export const getMessagesByContactId = query({
  args: {
    contactId: v.string(),
  },
  handler: async (ctx, args) => {
    const rows = await ctx.db
      .query("instagramConversations")
      .withIndex("by_contact_id", (q) => q.eq("contactId", args.contactId))
      .collect();

    const messages: { role: string; content: string }[] = [];
    for (const row of rows) {
      if (row.message) {
        messages.push({ role: "user", content: row.message });
      }
      if (row.agentReplyText) {
        messages.push({ role: "assistant", content: row.agentReplyText });
      }
    }

    return messages;
  },
});
