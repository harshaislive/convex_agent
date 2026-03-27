import { mutation } from "./_generated/server";
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
