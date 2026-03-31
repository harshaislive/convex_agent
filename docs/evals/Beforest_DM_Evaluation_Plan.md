# Beforest DM Evaluation Plan

## Goal
Run a 100-query evaluation against `POST /beforest/reply` to measure whether the Beforest agent is accurate, grounded, short-form, and commercially useful in Instagram-style conversations.

## Sources Of Truth
- Outline is the primary source of truth for factual brand information.
- Approved live domains are fallback sources for current public page details:
  - `beforest.co`
  - `*.beforest.co`
  - `bewild.life`

## Success Criteria
Score each response on a `pass` / `fail` basis for the following:

- `grounded_pass`: The answer is consistent with Outline or approved live pages.
- `routing_pass`: The answer routes to the correct Beforest property when a route is needed.
- `hallucination_pass`: The answer does not invent missing facts, pricing, availability, approvals, or internal policies.
- `dm_style_pass`: The answer fits Instagram DM style: short, clear, and easy to act on.
- `actionability_pass`: The answer gives the next useful step when relevant.
- `tone_pass`: The answer sounds mature, assertive, grounded, and not robotic or salesy.
- `link_discipline_pass`: The answer avoids unnecessary link spam and uses at most one relevant link by default.

## Global Rules
A strong answer should usually:
- stay within 1 to 3 short sentences
- stay under 320 characters unless the question truly needs more
- answer first, then give one next step if useful
- ask at most one follow-up question, only when needed
- prefer approved facts over marketing flourish

## Target Thresholds
- Groundedness: `>= 90/100`
- No hallucination: `>= 95/100`
- Correct routing: `>= 90/100`
- DM suitability: `>= 85/100`
- Actionability: `>= 85/100`
- Tone: `>= 85/100`

## Query Mix
- Beforest as a whole: 15
- Collectives: 20
- Visiting / joining / staying: 15
- Influencers / creators / coverage: 10
- Partnerships / collaborations: 10
- Event hosts / retreat facilitators / venue: 10
- Current experiences: 10
- Bewild produce / products: 5
- Routing / link requests: 3
- Unclear / edge cases: 2

## 100 Queries

### Beforest As A Whole (15)
1. What is Beforest?
2. What does Beforest do?
3. Can you explain Beforest in simple words?
4. What is the idea behind Beforest?
5. What makes Beforest different?
6. Is Beforest a brand, a place, or a community?
7. What is Beforest building?
8. What is Beforest about?
9. What does Beforest stand for?
10. Who is Beforest for?
11. What kind of lifestyle is Beforest promoting?
12. What all exists under Beforest?
13. Is Beforest only about farming?
14. Is Beforest a hospitality brand?
15. Can you give me a quick overview of Beforest?

### Collectives (20)
16. What is Hammiyala collective?
17. Tell me about Hammiyala collective.
18. Where is Hammiyala collective located?
19. What is life at Hammiyala like?
20. Is Hammiyala for visitors or residents?
21. What is Bhopal collective?
22. Tell me about Bhopal collective.
23. Where is the Bhopal collective?
24. What is Poomaale 2.0?
25. Is Poomaale a collective?
26. What are Beforest collectives?
27. How many collectives does Beforest have?
28. Which Beforest collective is in Coorg?
29. Which collective should I look at first if I want nature and community?
30. What is the purpose of a Beforest collective?
31. Are the collectives open to everyone?
32. Can families come to the collectives?
33. Are collectives the same as experiences?
34. Are collectives the same as stays?
35. Which Beforest collective should I explore if I want the planter lifestyle vibe?

### Visiting / Joining / Staying (15)
36. How do I join Hammiyala collective?
37. How do I become part of a Beforest collective?
38. Can I visit Hammiyala?
39. Can I stay at a Beforest collective?
40. How can I explore Beforest in person?
41. I want to visit Beforest. What should I do?
42. I want to stay for a few days. Where do I start?
43. Is there a way to book a visit?
44. Can I apply to join a collective?
45. How do I know which Beforest path is right for me?
46. I want to spend a weekend with Beforest. What should I explore?
47. Can I come with my family?
48. Can I come as a solo traveler?
49. Where do I see available stays?
50. What is the best way to start if I want to engage with Beforest offline?

### Influencers / Creators / Coverage (10)
51. I am a creator and would love to stay and make content. Who should I speak to?
52. I am an influencer. Can I collaborate with Beforest for a stay and coverage?
53. Do you host creators at your properties?
54. I make travel and lifestyle content. Can I partner with Beforest?
55. I want to visit and create reels around Beforest. Is that possible?
56. Can I do a barter stay in exchange for coverage?
57. I run a wellness page and want to cover your collective. What is the process?
58. I am a filmmaker and want to document Beforest. Who do I contact?
59. I want to feature Bewild and Beforest on my page. Can we collaborate?
60. I have an audience in sustainability and travel. How can I pitch a creator collaboration?

### Partnerships / Collaborations (10)
61. Does Beforest do brand partnerships?
62. I want to collaborate with Beforest. What is the process?
63. Do you explore strategic partnerships?
64. I have a regenerative brand and want to partner with Beforest.
65. I want to discuss a collaboration with your team.
66. Can we partner on a community or sustainability initiative?
67. I run a conscious brand. Is Beforest open to collaborations?
68. I want to pitch a partnership to Beforest. Where should I send it?
69. Do you work with aligned businesses and founders?
70. I want to explore a partnership across experiences and community. Can we talk?

### Event Hosts / Retreat Facilitators / Venue (10)
71. I host retreats. Can I host one at a Beforest collective?
72. Do you allow external facilitators to host workshops?
73. I want to organize a wellness retreat with Beforest. Is that possible?
74. Can your collective host a private event?
75. I am looking for a nature-led venue for a gathering. Can Beforest help?
76. I want to host a yoga retreat with you.
77. Do you host corporate offsites or curated group gatherings?
78. Can I host a learning or community event at your space?
79. I run immersive events. Can we collaborate on a hosted experience?
80. What is the process for hosting a retreat or event with Beforest?

### Current Experiences (10)
81. What experiences are currently live?
82. What can I book right now?
83. Are there any upcoming retreats or workshops?
84. Where do I see your latest experiences?
85. What is currently available on Beforest experiences?
86. Are there any Coorg experiences live right now?
87. Show me current Beforest experiences.
88. I want to browse all current experiences.
89. Do you have any current workshops or retreats?
90. Which website should I check for live experiences?

### Bewild Produce / Products (5)
91. What is Bewild?
92. Can I buy produce from Bewild?
93. Where do I browse Bewild products?
94. Is Bewild part of Beforest?
95. What kind of products does Bewild offer?

### Routing / Link Requests (3)
96. Send me the link for experiences.
97. Send me the link for products.
98. Where should I go if I want stays instead of experiences?

### Unclear / Edge Cases (2)
99. I want to work with you somehow. Where should I start?
100. Tell me everything about Beforest in detail.

## Expected Behavior By Bucket
- Beforest as a whole: clarify the brand simply and confidently.
- Collectives: explain what the collective is and avoid inventing process details.
- Visiting / joining / staying: give the next useful step and correct route.
- Influencers / creators: be open and professional, but do not promise approvals.
- Partnerships: acknowledge fit-based interest, ask for minimal next info if needed.
- Event hosts: identify whether this is a venue / retreat / facilitation ask and route appropriately.
- Current experiences: prefer live public listings over stale assumptions.
- Bewild: route clearly to `bewild.life` and distinguish products from stays or experiences.
- Edge cases: stay calm, concise, and grounded; do not dump long essays.

## Output Format
Recommended run output fields:
- `id`
- `category`
- `query`
- `response`
- `grounded_pass`
- `routing_pass`
- `hallucination_pass`
- `dm_style_pass`
- `actionability_pass`
- `tone_pass`
- `link_discipline_pass`
- `notes`

## CSV Guidance
Save one row per query. Use `pass` / `fail` for each criterion and keep reviewer notes short and specific.
