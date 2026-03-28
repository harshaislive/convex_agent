import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const appendMessage = mutation({
  args: {
    contactId: v.string(),
    role: v.string(),
    content: v.string(),
    timestamp: v.float64(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("signal_conversations")
      .withIndex("by_contact_id", (q) => q.eq("contactId", args.contactId))
      .first();

    const newMessage = { role: args.role, content: args.content };

    if (existing) {
      const messages = (existing.messages as Record<string, unknown>[] | undefined) || [];
      await ctx.db.patch(existing._id, {
        messages: [...messages, newMessage],
        lastMessageAt: args.timestamp,
        messageCount: (existing.messageCount || 0) + 1,
      });
    } else {
      await ctx.db.insert("signal_conversations", {
        contactId: args.contactId,
        createdAt: args.timestamp,
        lastMessageAt: args.timestamp,
        messageCount: 1,
        messages: [newMessage],
      });
    }
  },
});

export const appendInstagramStyleMessage = mutation({
  args: {
    contactId: v.string(),
    role: v.string(),
    content: v.string(),
    timestamp: v.float64(),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("instagramConversations", {
      contactId: args.contactId,
      message: args.role === "user" ? args.content : undefined,
      agentReplyText: args.role === "assistant" ? args.content : undefined,
      receivedAt: args.timestamp,
      agentReplied: args.role === "assistant",
      agentReplyAt: args.role === "assistant" ? args.timestamp : undefined,
      lastReplyType: args.role,
    });
  },
});

export const getMessages = query({
  args: {
    contactId: v.string(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("signal_conversations")
      .withIndex("by_contact_id", (q) => q.eq("contactId", args.contactId))
      .first();

    if (!existing) return [];
    return (existing.messages as Record<string, string>[]) || [];
  },
});
