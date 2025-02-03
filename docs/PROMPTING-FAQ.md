# Prompting FAQ

## My model isn’t returning the desired results for the question I trained it on in ilab model chat: what do I do?

Let’s use the following example: we are aligning a model on knowledge about a company Retro911. The example knowledge leaf node contains the following section:

```yaml
seed_examples:
  - context: |
      **Retro911** is a full-service company dedicated to helping enthusiasts navigate the world of retro video systems.
      Founded in 2024 by Grant Shipley and headquartered in Clayton, NC, it specializes in providing expert advice on
      conquering classic game titles, assisting in the acquisition and restoration of vintage hardware, and fostering
      a community passionate about all things retro gaming.
    questions_and_answers:
      - question: |
          When was Retro911 founded?
        answer: |
          Retro911 was founded in 2024.
      - question: |
          Who founded Retro911?
        answer: |
          Grant Shipley founded Retro911.
```

After aligning the model on this data: we proceed to chat with the model using ilab model chat. We ask it the following question and get the following response:

```text
>>> When was Retro911 founded?                                                                                                                                                                                                                                                                                                    [S][default]
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── /var/mnt/models/retro911model ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ I'm sorry, but I can't provide specific information about the founding date of Retro911. I recommend checking their official website or contacting them directly for accurate details.                                                                                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.470 seconds ─╯

```

Not the answer we were looking for: where do we go from here? The key is to provide more context to help the model answer the question. Let's look at a variation of the question asked below that produces the proper response:

```text
>>> I want to talk about Retro911 the full-service company dedicated to helping enthusiasts navigate the world of retro video systems. When was it founded?                                                                                                                                                                                                                                                                                                          [S][default]
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── /var/mnt/models/retro911model ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ I apologize for the confusion earlier. I had the wrong information in my system. Retro911 was founded in 2024. They started their operations in that year and have since become a well-known and respected name in the retro video gaming community.                                                                                       │
│                                                                                                                                                                                                                                                                                                                                            │
│ Retro911 has been dedicated to helping enthusiasts navigate the world of retro video systems, offering a wide range of services, such as repair and restoration of retro video systems, sourcing and acquisition of rare and collectible hardware, and hosting of events and conventions for retro gaming enthusiasts.                     │
│                                                                                                                                                                                                                                                                                                                                            │
│ To learn more about their services and what they have to offer, I recommend visiting their official website or reaching out to them directly. They're sure to have a lot of exciting information and events planned for the future!                                                                                                        │
│                                                                                                                                                                                                                                                                                                                                            │
│ Keep in mind that the specifics of their services and events may change over time, so it's best to check their latest updates for the most accurate information. Have fun exploring the world of retro gaming with Retro911!                                                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 2.737 seconds ─╯
```

Now we are able to get a valid response with the proper data by providing the model more context about the company we are gathering information on (beyond just the name). The additional context allows the model to focus on the proper details to report the appropriate information.