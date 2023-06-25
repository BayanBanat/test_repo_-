import tensorflow as tf
import numpy as np

# Define your dataset
dataset = [
  {
      "patterns": "I had a head injury a few years ago and my mind races all the time. I have trouble sleeping and have a lot of anxiety. Every medicine I have been on my body rejects; I get sick to my stomach and get blisters in my mouth. How can I calm my self down? I'm a wreck.",
      "responses":  "It sounds like you're dealing with the aftermath of a head injury. I recommend seeking medical advice to address your symptoms and explore alternative treatments for anxiety and sleep issues. A healthcare professional can provide personalized guidance and suggest techniques such as relaxation exercises or therapy to help you calm down."
  },
  {
      "patterns": "I want a secure relationship with someone that wants to be with me and who will actually put effort into it.I seem to gravitate toward unavailable men and those that want intimacy and no relationship. I let men dictate and control me because they accuse me of being controlling. I let men emotionally abuse me and I am at their beck and call.I am not comfortable being alone or doing anything by myself. I feel I need the security of someone being around just to survive. I know what I'm doing wrong and I do it anyway just hoping things will change. How do I stop this behavior and thought process?",
      "responses":"It's important to prioritize your well-being and seek healthy, fulfilling relationships. Recognizing the patterns you've described is an important first step. Consider working with a therapist or counselor who can help you explore your self-worth, boundaries, and self-reliance. They can guide you in developing healthier relationship dynamics and provide support as you work on changing these patterns."
  },
  {
      "patterns": "I easily recognize this but have no control over it and need suggestions for managing my anger.",
      "responses": "Managing anger can be challenging, but there are techniques you can try. Consider exploring anger management strategies such as deep breathing exercises, mindfulness meditation, or engaging in physical activities like exercise or sports. Additionally, seeking therapy or counseling can provide you with professional guidance tailored to your specific situation. A mental health professional can help you develop coping mechanisms and address any underlying issues contributing to your anger."
  },
  {
      "patterns": "It takes me a long time to fall asleep; I\u00e2\u20ac\u2122d estimate about two hours. I often have nightmares, starting with being eaten by a monster, and I often wake up frightened and unable to breathe.I believe I started losing sleep after breaking up with a girlfriend of 8 years. Also, my father\u00e2\u20ac\u2122s business went bankrupt and my mother has a chronic condition. I was under a lot of stress, and life lost meaning.After improving my life and developing a habit of running and exercise to release stress, my symptoms improved. However in the past couple months, I started losing sleep again and having nightmares without warning. How can I solve this issue?",
      "responses": "Sleep disturbances and nightmares can be linked to various factors, including stress and emotional challenges. Since you experienced an improvement by focusing on exercise and stress release in the past, it might be helpful to revisit those strategies. Additionally, consider establishing a consistent bedtime routine, creating a comfortable sleep environment, and practicing relaxation techniques before sleep. If the problem persists, it's advisable to consult a healthcare professional or a sleep specialist who can provide further guidance and support."
  },
  {
      "patterns": "What do you do when a therapist and a parent drugs a child and makes up lies in order to stop the other parent's custody visitation?",
      "responses": "I'm not a legal expert, but if you suspect illegal or harmful behavior, it's crucial to prioritize the well-being and safety of the child involved. It's advisable to consult with a family law attorney or child protective services to address this serious issue. They can guide you on appropriate steps to take and help protect the child's best interests."
  },
  {
      "patterns": "They're calling me names like hypocrite and a baby even when they act in the same manner as I do. I'm tired of being called names. What should I do?",
      "responses":  "Being subjected to name-calling can be hurtful and damaging. It's essential to prioritize your emotional well-being and establish boundaries. You can start by calmly expressing how their words affect you and requesting that they treat you with respect. If the name-calling persists or escalates, it might be helpful to distance yourself from individuals who engage in such behavior. Surrounding yourself with supportive and understanding people can contribute to a healthier and more positive environment."
  },
  {
      "patterns": "Cheating is something unacceptable for me but because we have two daughters I decided not to break up the family.However, now I am struggling to forget and forgive what happened. I feel like I cannot trust him. Without trust, I cannot stay in this relationship. On the other hand, I do not want my children to get hurt. I'm not sure how to move forward?",
      "responses": "Dealing with infidelity is a complex and challenging situation. Rebuilding trust in a relationship after cheating requires open communication, transparency, and both parties' willingness to work on the relationship. Couples counseling or therapy can provide a supportive environment to address the issues, explore the impact of the infidelity, and work towards healing. A professional can guide you through the process and help you make informed decisions about the future of your relationship."
  },
  {
      "patterns": "I have a lot of issues going on right now. First of all, I have a lot of trouble sleeping at times, while other nights I sleep too much and still feel quite tired. I\u00e2\u20ac\u2122m also noticing increased irritability and experiencing anxiety attacks that last for hours. Is there something wrong with me and if so what should I do?",
      "responses": "Experiencing sleep disturbances, changes in mood, and anxiety attacks can indicate underlying issues that could benefit from professional evaluation and support. It's advisable to consult a healthcare professional or mental health provider who can conduct a thorough assessment and provide appropriate guidance. They can help determine if there are any underlying medical conditions or psychological factors contributing to your symptoms and suggest treatment options tailored to your needs."
  },
  {
      "patterns": "I have no sex drive due to medical issues. I\u00e2\u20ac\u2122ve shut down completely and closed everyone out. I\u00e2\u20ac\u2122ve even told my husband that I wanted a divorce since that\u00e2\u20ac\u2122s what I thought he wanted to hear. However, it devastated me when I learned he's seeing someone else.I\u00e2\u20ac\u2122ve since told him that I wanted to work things out but he\u00e2\u20ac\u2122s not sure I actually mean it. He thinks the only reason I want him back is because I\u00e2\u20ac\u2122m jealous.  How do I show him I\u00e2\u20ac\u2122m serious?",
      "responses":  "Rebuilding trust and demonstrating your seriousness about working on the relationship can take time and consistent effort. Open communication is crucial in this situation. Express your genuine feelings and intentions to your husband, emphasizing your desire to work through the challenges together. Consider couples counseling or therapy as a safe space to address the issues, improve communication, and rebuild trust. A professional can guide both of you through this process and facilitate productive discussions."
  },
  {
      "patterns": "I have a lot of issues going on right now. First of all, I have a lot of trouble sleeping at times, while other nights I sleep too much and still feel quite tired. I\u00e2\u20ac\u2122m also noticing increased irritability and experiencing anxiety attacks that last for hours. Is there something wrong with me and if so what should I do?",
      "responses": "Experiencing sleep disturbances, mood changes, and anxiety attacks can be indicators of underlying issues. It's essential to prioritize your well-being and seek professional help. Consulting with a healthcare professional or mental health provider can assist in understanding the underlying causes and developing an appropriate treatment plan. They can offer guidance, support, and potential interventions to address your symptoms effectively."
  },
  {
      "patterns": "My sibling and I constantly argue and can't seem to find common ground. How can we resolve our conflicts and build a healthier sibling relationship?",
      "responses": "In order to resolve conflicts with your sibling and build a healthier relationship, it's important to communicate openly and honestly. Try to understand each other's perspectives, actively listen, and find common ground. It can also be helpful to establish boundaries, practice forgiveness, and seek professional help if needed."
  },
  {
      "patterns": "I'm facing financial difficulties and can't keep up with bills and expenses. What resources or strategies can I use to manage my finances more effectively and overcome this challenging situation?",
      "responses": "When facing financial difficulties, it's crucial to assess your expenses, create a budget, and prioritize essential needs. Look for ways to increase your income, such as exploring new job opportunities or freelance work. Seek guidance from financial experts, consider debt consolidation or negotiation, and explore community resources for financial assistance."
  },
  {
      "patterns": "I recently lost a loved one, and the grief is consuming me. How can I navigate the grieving process and find ways to honor their memory?",
      "responses": "Losing a loved one is a difficult experience. It's important to allow yourself to grieve and process your emotions. Seek support from friends, family, or grief support groups. Find healthy ways to remember and honor your loved one, such as creating a memorial or participating in activities that hold significance to them. Consider seeking professional counseling to navigate through the grieving process."
  },
  {
      "patterns": "I've been experiencing low self-esteem and a negative body image. What steps can I take to improve my self-confidence and develop a healthier relationship with my body?",
      "responses": "Having low self-esteem and a negative body image can be challenging, but there are steps you can take to improve your self-confidence. Practice self-care, engage in positive self-talk, focus on your strengths, surround yourself with supportive people, and consider seeking therapy or counseling to address underlying issues and develop a healthier relationship with your body."
  },
  {
      "patterns": "I'm in a toxic friendship, but I'm unsure how to end it without causing further harm. How can I gracefully distance myself and prioritize my well-being?",
      "responses": "Ending a toxic friendship can be difficult, but prioritizing your well-being is important. Set boundaries, communicate your feelings honestly but respectfully, gradually reduce contact, and focus on cultivating healthy relationships. Surround yourself with supportive friends and consider seeking professional guidance to navigate this process and heal from the toxic dynamics."
  },
  {
      "patterns": "My brother just broke up with his girlfriend. While they were dating for the past two years, her son spent the weekends with me and my mother. Now she doesn't think her son should have any interaction with us. How should we approach the situation? Is she doing the right thing for her child?",
      "responses": "The breakup can be a challenging time for everyone involved, including the child. It's essential to approach the situation with empathy and understanding. Respect the mother's decision regarding her child's interactions but express your willingness to maintain a positive relationship if it aligns with her wishes. Focus on providing emotional support to your brother during this time and ensure the child's well-being remains the top priority."
  },
  {
      "patterns": "My wife and mother are having tense disagreements. In the past, they've had minor differences. For example, my wife would complain to me my mother is too overbearing; my mother would complain my wife is lazy. However, it's intensified lately. I think the cause is my wife talked back to her once. Now, any little disagreement is magnified, leading to major disagreements. What can I do?",
      "responses": "Tense disagreements between your wife and mother can put you in a difficult position, but there are steps you can take to address the situation. Encourage open and respectful communication between them, and facilitate discussions to address any underlying issues. Set boundaries, express your feelings neutrally, and consider family therapy to help navigate the conflicts, foster understanding, and find constructive solutions."
  },
  {
      "patterns": "I'm planning to have a baby, so I have to quit smoking - but it's hard. Sometimes it's not a physical need; it's mental. I cannot help myself from thinking about smoking. What can I do to get rid of this addiction?",
      "responses": "Quitting smoking can be challenging, but there are strategies to help you overcome the addiction. Seek professional help from healthcare providers or smoking cessation programs. Create a support system of friends and family who can provide encouragement. Replace smoking with healthier habits and coping mechanisms, such as exercise or mindfulness techniques. Consider therapy or counseling to address the underlying mental and emotional aspects of the addiction."
  },
  {
      "patterns": "I have secrets in my mind, and I don't know what to do with them. I don't want to tell my wife and mom because I don't want to hurt them. But I'm not sure how long I can keep the secret to myself. What should I do? It's becoming annoying and making me anxious. Help me out",
      "responses": "Keeping secrets can be burdensome and contribute to anxiety. It's important to find a safe space to express yourself. Consider seeking professional help, such as therapy or counseling, to discuss your concerns confidentially and gain support in navigating this situation. A professional can provide guidance on when and how to address the secrets and help you manage the emotional impact they have on you."
  },
  {
      "patterns": "I am extremely possessive in my relationships, and this is hurting my friendships. How can I fix my underlying issues?",
      "responses": "Addressing possessiveness in relationships requires self-reflection and self-improvement. Recognize and understand the root causes of your possessiveness, such as insecurities or past experiences. Work on building self-confidence, practicing trust, and developing a healthy sense of independence. Communicate openly with your friends about your feelings and concerns, and seek support from a therapist or counselor to work through your underlying issues."
  },
  {
      "patterns": "I've been in a long-term relationship, but lately, my partner seems distant and uninterested. How can I communicate my concerns and work on improving our connection?",
      "responses": "Communication is key when addressing concerns in a long-term relationship. Find a suitable time to have an open and honest conversation with your partner. Express your feelings and concerns without blaming or criticizing them. Listen actively to their perspective and work together to find solutions. Consider relationship counseling or therapy to improve communication, rebuild connection, and strengthen your relationship."
  },
  {
      "patterns": "I'm a single parent and struggling to balance work and family responsibilities. What strategies can I use to create a healthy work-life balance and still be there for my children?",
      "responses": "Balancing work and family as a single parent can be challenging, but there are strategies to help create a healthy work-life balance. Prioritize and delegate tasks, set realistic expectations, and establish boundaries between work and family time. Seek support from family, friends, or support groups. Explore flexible work options and communicate openly with your employer. Remember to practice self-care and seek help when needed."
  },
  {
      "patterns": "I'm experiencing burnout at work, and it's affecting my mental health. How can I overcome burnout and restore a sense of well-being?",
      "responses": "Burnout can significantly impact mental health, but there are steps you can take to overcome it. Take time off to rest and recharge. Evaluate and prioritize tasks, set boundaries, and learn to say no. Engage in self-care activities, such as exercise, hobbies, or mindfulness practices. Seek support from colleagues, friends, or professional counselors. Consider talking to your employer about workload or seeking new job opportunities if necessary."
  },
  {
      "patterns": "My parents recently got divorced, and I feel torn between them. How can I manage the conflict and maintain a good relationship with both of them?",
      "responses": "Divorce can be challenging for children, and it's important to prioritize your well-being. Communicate openly with your parents about your feelings and concerns. Maintain a neutral position and avoid taking sides. Seek support from friends, family, or a therapist to help navigate your emotions. Set boundaries when discussing the divorce and focus on building individual relationships with each parent based on understanding and respect."
  },
  {
      "patterns": "I suspect that a close friend is struggling with addiction, but they haven't opened up about it. How can I approach the topic sensitively and offer support?",
      "responses": "Approaching the topic of addiction requires sensitivity and care. Choose a private and non-judgmental setting to talk to your friend. Express your concerns from a place of compassion and offer your support. Avoid pushing them to disclose their struggles but let them know you are there to listen and assist if needed. Suggest resources such as helplines or support groups, and encourage them to seek professional help when they are ready."
  },
  {
      "patterns": "I've been feeling overwhelmed with stress and anxiety lately. What coping mechanisms or self-care practices can I incorporate into my daily routine to improve my mental well-being?",
      "responses": "Managing stress and anxiety is essential for your mental well-being. Consider incorporating practices like regular exercise, deep breathing exercises, mindfulness or meditation, journaling, and maintaining a balanced diet. Ensure you're getting enough sleep and prioritize self-care activities that bring you joy. Don't hesitate to seek professional help, such as therapy or counseling, to develop personalized coping strategies and address underlying causes."
  },
  {
      "patterns": "My best friend betrayed my trust. How can I rebuild the friendship or move on from the situation?",
      "responses": "Rebuilding trust after a betrayal can be challenging but not impossible. Take time to process your emotions and assess whether the friendship is worth salvaging. Communicate your feelings honestly and openly with your friend, expressing the impact of their actions. Set clear boundaries moving forward and observe if they make efforts to rebuild trust. However, it's also important to recognize when it's healthier to let go and focus on nurturing relationships with people who prioritize your trust and well-being."
  },
  {
      "patterns": "I'm feeling overwhelmed with financial problems. What strategies can I use to manage my finances and reduce stress?",
      "responses": "Financial problems can cause significant stress, but there are steps you can take to manage them. Start by creating a budget and tracking your expenses. Prioritize essential expenses and look for areas where you can cut back. Explore options for increasing income or finding additional sources of support. Consider seeking advice from a financial counselor or educator to create a plan for debt management and long-term financial stability."
  },
  {
      "patterns": "I recently moved to a new city, and I'm finding it hard to make friends. How can I build a social network and feel connected?",
      "responses": "Building a social network in a new city takes time and effort. Start by exploring local community groups, clubs, or hobby-based organizations that align with your interests. Attend events or join social platforms focused on connecting people with similar hobbies or backgrounds. Be open to new experiences and initiate conversations with others. Volunteering or taking classes can also provide opportunities to meet like-minded individuals. Remember, building friendships takes time, so be patient and persistent."
  },
  {
      "patterns": "I'm dealing with a toxic friendship. How can I establish healthy boundaries and maintain my well-being?",
      "responses": "Dealing with a toxic friendship requires establishing healthy boundaries. Evaluate the impact of the friendship on your well-being and consider reducing contact or ending the relationship if necessary. Communicate your boundaries clearly and assertively, and be prepared for potential resistance or backlash. Seek support from trusted friends or a therapist to help you navigate this process. Focus on nurturing relationships that bring positivity and support to your life."
  },
  {
      "patterns": "I'm experiencing a loss of motivation and interest in my hobbies. How can I regain my passion and find joy again?",
      "responses": "Experiencing a loss of motivation and interest in hobbies is common. Start by exploring new activities or revisiting old ones that used to bring you joy. Set small, achievable goals to regain a sense of accomplishment. Connect with others who share similar interests, as their enthusiasm can be contagious. Take breaks when needed and practice self-compassion. Reflect on the underlying causes of the loss of passion and consider seeking professional help if it persists."
  },
  {
      "patterns": "I'm facing a major life transition. I just changed my career. How can I adapt to the changes and navigate this new chapter?",
      "responses": "Changing careers can be both exciting and challenging. Embrace the opportunity to learn and grow in your new field. Be patient with yourself as you adjust to the changes and set realistic expectations. Seek support from mentors or professionals in your new career path. Upgrade your skills if necessary through courses or training programs. Maintain a positive mindset and stay open to new possibilities that may arise during this transitional period."
  },
  {
      "patterns": "How can I set healthy boundaries with others to protect my well-being?",
      "responses": "Setting healthy boundaries is crucial for protecting your well-being. Clearly define your limits and communicate them assertively but respectfully. Be consistent in enforcing your boundaries and expect some resistance or pushback initially. Surround yourself with people who respect your boundaries and offer support. Remember that setting boundaries is not selfish; it is an act of self-care that promotes healthy relationships."
  },
  {
      "patterns": "What are some effective stress management techniques I can implement?",
      "responses": "There are various effective stress management techniques you can implement. Some examples include regular exercise, deep breathing exercises, practicing mindfulness or meditation, maintaining a healthy diet, getting enough sleep, engaging in hobbies or activities you enjoy, seeking social support, and setting aside time for relaxation and self-care. Experiment with different techniques to find what works best for you, and consider seeking professional help if stress becomes overwhelming."
  },
  {
      "patterns": "I recently lost my job and I'm feeling hopeless. How can I stay motivated and find new opportunities?",
      "responses": "Losing a job can be tough, but it's important to stay motivated and proactive. Here are some steps you can take:\n1. Take time to process your emotions and give yourself permission to grieve the loss.\n2. Create a routine to maintain structure and stay productive. Set goals and create a schedule.\n3. Update your resume, polish your LinkedIn profile, and start networking. Reach out to friends, colleagues, and professional contacts for job leads and recommendations.\n4. Consider acquiring new skills or certifications to make yourself more marketable.\n5. Stay positive and persistent. Rejections are a normal part of the process, so don't get discouraged. Keep applying and attending job fairs or networking events.\n6. Take care of your physical and mental well-being. Exercise, eat healthily, and seek support from friends and family.\nRemember, this is an opportunity for growth and new beginnings. Stay motivated, and you'll find new opportunities."
   },
  {
      "patterns": "My parents are going through a divorce and it's affecting me deeply. How can I cope with the emotional turmoil?",
      "responses": "Dealing with your parents' divorce can be emotionally challenging, but there are ways to cope:\n1. Allow yourself to feel your emotions and express them in a healthy way. Find someone you trust to talk about your feelings, such as a friend, family member, or therapist.\n2. Take care of your physical well-being by maintaining a regular sleep schedule, eating nutritious meals, and engaging in physical activity.\n3. Seek support from others who have experienced similar situations, such as support groups or online communities.\n4. Maintain open communication with your parents and express your needs and concerns. It's important to have your voice heard.\n5. Engage in activities that bring you joy and provide a sense of normalcy and stability.\n6. Consider seeking professional help, such as therapy, to help you navigate through this challenging time.\nRemember, it's normal to feel a range of emotions, but with time and support, you can find healing and resilience."
  },
  {
      "patterns": "I'm having difficulties in my romantic relationship. What steps can I take to improve communication and resolve conflicts?",
      "responses": "Improving communication and resolving conflicts in a relationship takes effort and commitment. Here are some steps you can take:\n1. Practice active listening by giving your full attention to your partner and seeking to understand their perspective without interrupting.\n2. Use \"I\" statements to express your feelings and needs instead of blaming or accusing your partner.\n3. Avoid defensive or aggressive responses during conflicts. Take a step back, breathe, and try to approach the situation with empathy.\n4. Find a time when both of you are calm to discuss the issues. Use a constructive approach, focusing on problem-solving rather than winning arguments.\n5. Seek professional help, such as couples therapy, if the issues persist or if you need guidance in improving your communication skills.\nRemember, building a healthy relationship takes time and effort from both partners. Open and honest communication is key."
  },
  {
      "patterns": "I'm struggling with a decision and I don't know what to do. How can I gain clarity and make the right choice?",
      "responses": "When faced with a difficult decision, it's natural to feel unsure. Here are some strategies to help you gain clarity and make the right choice:\n1. Gather information about each option. Consider the potential outcomes and consequences of each choice.\n2. Reflect on your values, goals, and priorities. Which option aligns best with what you want to achieve in the long run?\n3. Consider seeking advice from trusted friends, family members, or mentors who have experience or knowledge in the relevant area.\n4. Take a step back and give yourself some time and space to think. Sometimes, stepping away from the decision momentarily can bring new insights.\n5. Trust your intuition. Pay attention to your gut feelings and inner wisdom.\n6. Consider journaling or creating pros and cons lists to organize your thoughts.\nRemember, there's no foolproof method for making decisions, but by gathering information, reflecting on your values, and trusting yourself, you can increase your chances of making the right choice."
  },
  {
      "patterns": "How can I deal with feelings of loneliness and isolation?",
      "responses": "Feeling lonely and isolated can be challenging, but there are ways to cope and overcome these emotions:\n1. Reach out to others. Connect with friends, family, or support groups either in person or through virtual platforms.\n2. Engage in activities you enjoy or explore new hobbies. This can help you meet like-minded people and create a sense of belonging.\n3. Practice self-care and self-compassion. Take time for yourself, engage in relaxation techniques, and prioritize your well-being.\n4. Volunteer or get involved in your community. Helping others can create a sense of connection and purpose.\n5. Consider seeking professional help if feelings of loneliness and isolation persist or if they significantly impact your daily life.\nRemember, you are not alone, and there are people who care and want to support you. Reach out and stay connected."
  },
  {
      "patterns": "I'm struggling with low self-esteem. What are some techniques to build self-confidence?",
      "responses": "Building self-confidence takes time and practice. Here are some techniques that can help:\n1. Challenge negative self-talk and replace it with positive affirmations. Focus on your strengths and achievements.\n2. Set realistic goals and celebrate small victories along the way. Each accomplishment boosts your confidence.\n3. Surround yourself with supportive and positive people who uplift and encourage you.\n4. Take care of your physical and mental well-being through exercise, proper nutrition, and self-care activities.\n5. Practice self-compassion and embrace self-acceptance. Treat yourself with kindness and forgive yourself for mistakes.\n6. Step out of your comfort zone and face your fears. Each new experience can contribute to your personal growth and confidence.\nRemember, building self-confidence is a journey. Be patient with yourself and celebrate your progress along the way."
  },
  {
      "patterns": "How do I cope with the loss of a loved one?",
      "responses": "Coping with the loss of a loved one is a deeply personal process. Here are some strategies that may help:\n1. Allow yourself to grieve and process your emotions in your own way and at your own pace. There is no right or wrong way to mourn.\n2. Seek support from family, friends, or support groups who have experienced similar losses. Talking about your feelings can provide comfort and understanding.\n3. Take care of yourself physically by maintaining a regular sleep schedule, eating nutritious meals, and engaging in exercise or activities that bring you solace.\n4. Create rituals or find ways to honor and remember your loved one. This can include writing in a journal, visiting their grave, or creating a memorial.\n5. Consider seeking professional help, such as grief counseling or therapy, to navigate the grieving process.\nRemember, healing takes time, and it's important to be patient and gentle with yourself as you navigate through this difficult time."
  },
  {
      "patterns": "What are some ways to manage and reduce anxiety in daily life?",
      "responses": "Managing and reducing anxiety can involve various strategies. Here are some techniques that may help:\n1. Practice deep breathing exercises or meditation to promote relaxation and calmness.\n2. Engage in regular physical exercise, which can help reduce anxiety and improve your overall well-being.\n3. Identify and challenge negative thoughts or cognitive distortions that contribute to anxiety. Replace them with more positive and realistic thoughts.\n4. Establish a consistent sleep routine to ensure you're getting enough rest, as lack of sleep can increase anxiety.\n5. Prioritize self-care activities that bring you joy and relaxation, such as hobbies, spending time in nature, or practicing mindfulness.\n6. Consider seeking therapy or counseling to learn additional coping skills and techniques tailored to your specific needs.\nRemember, everyone's experience with anxiety is unique, so it may take some trial and error to find the strategies that work best for you."
  },
  {
      "patterns": "How can I improve my communication skills in personal relationships?",
      "responses": "Improving communication skills can enhance personal relationships. Here are some tips to consider:\n1. Practice active listening. Give your full attention to the speaker, maintain eye contact, and avoid interrupting. Show empathy and validate their feelings.\n2. Express yourself clearly and assertively. Use \"I\" statements to communicate your thoughts and feelings without blaming or criticizing.\n3. Pay attention to nonverbal cues, such as body language and tone of voice. These can convey messages that words alone may not capture.\n4. Practice conflict resolution skills. Learn to compromise, find common ground, and seek win-win solutions.\n5. Be open to feedback and willing to apologize when necessary. Admitting mistakes and taking responsibility can strengthen relationships.\n6. Seek opportunities for relationship-building activities, such as spending quality time together and engaging in shared interests.\nRemember, communication is a two-way process that requires active participation and mutual understanding."
  },
  {
      "patterns": "I'm going through a major life transition. How do I navigate this change and maintain a positive outlook?",
      "responses": "Navigating major life transitions can be challenging, but maintaining a positive outlook can help. Here's how:\n1. Acknowledge and accept your emotions. It's natural to experience a range of feelings during transitions, including uncertainty and anxiety.\n2. Focus on the opportunities and growth that the transition can bring. Embrace change as a chance for personal development and new experiences.\n3. Set realistic goals and break them down into manageable steps. This can help you stay focused and motivated during the transition.\n4. Seek support from loved ones or a mentor who can provide guidance and encouragement.\n5. Practice self-care to ensure your well-being during the transition. Prioritize activities that reduce stress and promote relaxation.\n6. Cultivate a positive mindset by practicing gratitude and reframing challenges as opportunities for learning and growth.\nRemember, transitions are a natural part of life, and with time, patience, and a positive mindset, you can navigate them successfully."
  },
  {
      "patterns": "I've been experiencing sleep problems and it's affecting my overall well-being. Any tips for improving sleep quality and establishing a bedtime routine?",
      "responses": "Improving sleep quality and establishing a bedtime routine can contribute to overall well-being. Consider these tips:\n1. Maintain a consistent sleep schedule by going to bed and waking up at the same time each day, even on weekends.\n2. Create a calming bedtime routine to signal your body that it's time to sleep. This can include activities like reading, taking a warm bath, or practicing relaxation techniques.\n3. Make your sleep environment comfortable and conducive to sleep. Ensure your bedroom is cool, dark, and quiet. Consider using earplugs, eye masks, or white noise machines if needed.\n4. Limit exposure to electronic devices, such as smartphones and computers, before bed. The blue light emitted by screens can interfere with sleep.\n5. Avoid consuming caffeine, nicotine, and large meals close to bedtime, as these can disrupt sleep.\n6. Engage in regular physical exercise, but try to finish your workout at least a few hours before bedtime.\nRemember, it may take time to establish a consistent sleep routine, so be patient and persistent in implementing these strategies."
  },

  {
      "patterns": "I feel overwhelmed and stressed with my workload. How can I manage my tasks effectively?",
      "responses": "Managing tasks effectively can help reduce overwhelm and stress. Consider these strategies: Prioritize tasks, break them down, create a schedule, delegate when possible, practice time management, learn to say no, and take care of yourself."
  },
  {
      "patterns": "I'm having trouble sleeping at night. What are some strategies to improve my sleep quality?",
      "responses": "Improving sleep quality can involve various strategies. Here are some tips: Establish a bedtime routine, create a sleep-friendly environment, limit electronic devices, avoid stimulants, engage in regular exercise, manage stress, and stick to a sleep schedule."
  },
  {
      "patterns": "I'm a perfectionist and it's causing a lot of stress and self-criticism. How can I overcome perfectionism and learn to embrace imperfection?",
      "responses": "Perfectionism can be challenging. To overcome it, consider these strategies: Set realistic expectations, practice self-compassion, focus on progress rather than perfection, challenge negative self-talk, seek support from others, and celebrate small successes."
  },
  {
      "patterns": "I'm a caregiver for a loved one and feeling burnt out. Any suggestions for self-care and preventing caregiver fatigue?",
      "responses": "As a caregiver, self-care is crucial. Here are some suggestions: Set boundaries and ask for help, prioritize self-care activities, seek support from others, maintain a healthy lifestyle, take breaks and rest when needed, and consider joining a caregiver support group."
  },
  {
      "patterns": "I often feel overwhelmed by my emotions and have trouble regulating them. Any strategies for managing and expressing my emotions in a healthy way?",
      "responses": "Managing and expressing emotions is important. Try these strategies: Identify and label your emotions, practice deep breathing and relaxation techniques, journal about your feelings, engage in physical activity, seek professional help if needed, and consider talking to a trusted friend or therapist."
  },
  {
      "patterns": "I'm dealing with a major life change and feeling scared of the unknown. How can I embrace change and find courage to move forward?",
      "responses": "Major life changes can be intimidating. Here's how you can embrace change: Accept your feelings, focus on what you can control, break the change into smaller steps, seek support from loved ones, visualize positive outcomes, and remember that change often brings growth and new opportunities."
  },
  {
      "patterns": "I'm having difficulty maintaining healthy relationships. How can I improve my communication skills and build stronger connections with others?",
      "responses": "Improving communication skills can strengthen relationships. Consider these tips: Practice active listening, express yourself clearly and assertively, show empathy and understanding, be open to feedback, resolve conflicts respectfully, and invest time and effort into nurturing your relationships."
  },
  {
      "patterns": "I'm struggling with low self-esteem and negative self-talk. Any advice on building self-confidence and developing a positive mindset?",
      "responses": "Building self-confidence and a positive mindset takes time. Here are some strategies: Challenge negative thoughts, focus on your strengths and achievements, practice self-care and self-compassion, set realistic goals, surround yourself with supportive people, and engage in activities that boost your confidence."
  },
  {
      "patterns": "I have a hard time forgiving others who have hurt me in the past. How can I let go of resentment and move forward?",
      "responses": "Forgiveness can be difficult, but it's liberating. Try these steps: Acknowledge your emotions, understand the situation from different perspectives, practice empathy and compassion, set boundaries, consider therapy or counseling, and focus on your own healing and growth."
  },
  {
      "patterns": "I've been feeling unmotivated and uninspired lately. Any tips for finding motivation and reigniting my passion?",
      "responses": "Finding motivation and reigniting passion is possible. Try these strategies: Set meaningful goals, break them into smaller tasks, find inspiration from others, explore new interests, take breaks and practice self-care, celebrate progress, and remind yourself of your values and purpose."
  },
  {
      "patterns": "I'm constantly worrying about the future and it's affecting my daily life. How can I manage my anxiety and find peace of mind?",
      "responses": "Managing anxiety and finding peace of mind is essential. Consider these techniques: Practice mindfulness and meditation, challenge anxious thoughts, engage in relaxation exercises, establish a worry time, maintain a healthy lifestyle, seek professional help if needed, and prioritize self-care."
  },

  {
      "patterns": "I've been experiencing symptoms of burnout and exhaustion. Any strategies for preventing burnout and practicing self-care?",
      "responses": "When dealing with burnout and exhaustion, it's important to prioritize self-care. Some strategies you can try include setting boundaries, practicing relaxation techniques, maintaining a healthy work-life balance, getting enough sleep, exercising regularly, and seeking support from friends, family, or professionals."
  },
  {
      "patterns": "I have a friend who is struggling with addiction. How can I support them on their journey to recovery?",
      "responses": "Supporting a friend in their journey to recovery from addiction can involve offering a listening ear, being non-judgmental, encouraging professional help or support groups, and helping them establish healthy routines and coping mechanisms. It's also important to take care of yourself and set boundaries to avoid enabling their addictive behaviors."
  },
  {
      "patterns": "I'm going through a major life transition and feeling lost. How can I navigate this period of change and find a sense of direction?",
      "responses": "Navigating major life transitions can be challenging. Take time for self-reflection, set realistic goals, seek support from loved ones or professionals, explore new interests or hobbies, and consider seeking guidance through therapy or counseling to help find a sense of direction and adapt to the changes."
  },
  {
      "patterns": "I'm dealing with body image issues and low self-confidence. Any suggestions for developing a positive body image and self-acceptance?",
      "responses": "Developing a positive body image and self-acceptance takes time and self-compassion. Focus on self-care, surround yourself with positive influences, challenge negative thoughts, practice gratitude, engage in activities that make you feel good, and consider seeking support from therapists or support groups that specialize in body image and self-esteem."
  },
  {
      "patterns": "I feel lonely and isolated. How can I build a supportive social network and make meaningful connections?",
      "responses": "Building a supportive social network takes effort and patience. Start by joining social or hobby groups, volunteering, attending events or classes, and reaching out to friends or acquaintances. Engage in active listening, be open-minded, and invest time and energy in nurturing relationships that feel meaningful and supportive."
  },
  {
      "patterns": "I'm a parent and feeling guilty about not spending enough time with my children. How can I balance parenting and other responsibilities?",
      "responses": "Balancing parenting and other responsibilities can be challenging. Prioritize quality over quantity, create a flexible schedule, involve your children in age-appropriate tasks, practice self-care, communicate openly with your children, and seek support from your partner, family, or friends to help manage your responsibilities and alleviate guilt."
  },
  {
      "patterns": "I've been experiencing panic attacks and don't know how to manage them. Any tips for coping with panic disorder?",
      "responses": "Coping with panic attacks and panic disorder may involve deep breathing exercises, practicing mindfulness or meditation, challenging negative thoughts, gradually exposing yourself to triggers, seeking professional help for therapy or medication if needed, and learning relaxation techniques. It's important to consult with a mental health professional for personalized guidance and support."
  },
  {
      "patterns": "I have trouble asserting myself and saying no to others. How can I develop better boundaries and prioritize my own needs?",
      "responses": "Developing better boundaries and prioritizing your needs requires self-awareness and practice. Start by identifying your values and setting clear personal boundaries. Learn to communicate assertively, say no when necessary, delegate tasks, practice self-care, and seek therapy or counseling if needed to explore underlying causes and develop healthy assertiveness."
  },
  {
      "patterns": "I'm struggling with a recent breakup and finding it hard to move on. Any advice on healing a broken heart?",
      "responses": "Healing from a breakup takes time and self-care. Allow yourself to grieve and process your emotions. Lean on your support system, engage in activities you enjoy, focus on self-improvement, practice self-compassion, and consider seeking professional support through therapy or counseling to help navigate the healing process."
  },
  {
      "patterns": "I'm feeling overwhelmed with work and constantly stressed. How can I achieve a better work-life balance?",
      "responses": "Achieving a better work-life balance starts with setting boundaries and prioritizing self-care. Learn to delegate tasks, practice effective time management, establish a routine, take breaks, engage in stress-reducing activities, and communicate openly with your employer or colleagues about your needs. Remember that maintaining a healthy balance is crucial for your well-being."
  },
  {
      "patterns": "I'm a student facing exam stress and pressure. How can I manage my time effectively and reduce exam anxiety?",
      "responses": "Managing time effectively and reducing exam anxiety as a student requires planning and self-care. Create a study schedule, break tasks into manageable chunks, practice active learning techniques, prioritize self-care activities like exercise and sleep, seek support from teachers or classmates, and consider using relaxation techniques such as deep breathing or mindfulness exercises to manage exam stress."
  },
  {
      "patterns": "I've been dealing with procrastination and lack of motivation. Any strategies for increasing productivity and staying focused?",
      "responses": "Overcoming procrastination and increasing productivity starts with understanding underlying causes and implementing effective strategies. Break tasks into smaller, manageable steps, set specific goals, eliminate distractions, create a conducive work environment, establish a routine, practice time-blocking techniques, reward yourself for progress, and seek accountability through a study buddy or mentor."
  },

  {
      "patterns": "I've been experiencing relationship issues with my partner. How can we improve communication and strengthen our bond?",
      "responses": "Improving communication and strengthening your bond with your partner requires open and honest dialogue. Practice active listening, validate each other's feelings, use \"I\" statements to express your needs, and be willing to compromise. Consider couples therapy or relationship counseling to gain additional guidance and tools for improving your relationship."
  },
  {
      "patterns": "I have a friend who seems to be struggling with depression. How can I support them and encourage them to seek help?",
      "responses": "Supporting a friend with depression involves being there for them, listening without judgment, and encouraging them to seek professional help. Offer your support and let them know they're not alone. Help them research therapists or mental health resources, and consider reaching out to a mental health professional yourself for guidance on how to best support your friend."
  },
  {
      "patterns": "I'm feeling stuck in my career and unsure about my future. Any guidance on finding my passion and making meaningful career choices?",
      "responses": "Finding your passion and making meaningful career choices can be a journey of self-discovery. Reflect on your interests, values, and strengths. Explore different industries or job roles through internships, volunteering, or informational interviews. Consider working with a career counselor or coach who can provide guidance and support in navigating career decisions."
  },
  {
      "patterns": "I recently lost a loved one, and I'm struggling with grief. How can I cope with the loss and start the healing process?",
      "responses": "Coping with the loss of a loved one takes time and self-compassion. Allow yourself to grieve and honor your emotions. Seek support from friends, family, or support groups. Take care of your physical and emotional well-being through self-care activities. Consider seeking therapy or counseling to help navigate the grieving process and find ways to remember and honor your loved one."
  },
  {
      "patterns": "My parents don't understand me and constantly put pressure on me. How can I communicate with them effectively and set boundaries?",
      "responses": "Communicating with your parents effectively and setting boundaries requires open and respectful dialogue. Choose a calm and appropriate time to express your feelings and concerns. Use \"I\" statements to express your needs and avoid blaming language. Seek understanding and compromise, and consider involving a mediator or family therapist to facilitate the conversation if needed."
  },
  {
      "patterns": "I'm having trouble sleeping and often experience nightmares. Any suggestions for improving my sleep quality?",
      "responses": "Improving sleep quality starts with practicing good sleep hygiene. Create a bedtime routine, ensure your sleep environment is comfortable and free from distractions, limit caffeine and electronics before bed, and establish a consistent sleep schedule. If problems persist, consider consulting with a healthcare professional or sleep specialist for further evaluation and guidance."
  },
  {
      "patterns": "How can I improve my self-esteem and feel more confident in social situations?",
      "responses": "Improving self-esteem and feeling more confident in social situations takes time and self-compassion. Focus on your strengths and celebrate your achievements. Practice self-care, challenge negative self-talk, set realistic goals, and step out of your comfort zone gradually. Consider seeking support from a therapist or counselor who can provide additional guidance and strategies."
  },
  {
      "patterns": "I'm constantly feeling anxious and overwhelmed. What can I do to manage my anxiety?",
      "responses": "Managing anxiety involves developing coping strategies and seeking professional help if needed. Practice relaxation techniques such as deep breathing or meditation, engage in regular physical exercise, maintain a balanced lifestyle with healthy habits, challenge negative thoughts, and consider therapy or counseling to learn additional tools for managing anxiety."
  },
  {
      "patterns": "My wife just last week said she wants a divorce and it's all sudden. I understand that marriages have their ups and downs but I don't understand why she can't tell me why she wants a divorce. Should I fight for my family (daughter and wife)?",
      "responses": "Facing a potential divorce can be incredibly challenging. It's important to communicate openly with your wife and express your desire to understand her perspective. Consider seeking couples therapy to facilitate productive conversations and explore possible solutions. Ultimately, the decision to fight for your family or consider alternative paths depends on the specific circumstances and the willingness of both parties to work on the relationship."
  },
  {
      "patterns": "I loved him more than anything. He passed away on our anniversary, which was also the day I was going to apply for our apartment. I'm not handling it well. I moved to a new town and started a new life, but nothing helps.",
      "responses": "Losing someone you love can be devastating, and the grieving process is unique for each individual. It's important to give yourself time and space to mourn and heal. Reach out to friends, family, or a support group for comfort and understanding. Consider seeking therapy or counseling to help navigate the grieving process and find healthy ways to honor your loved one's memory."
  },
  {
      "patterns": "I did some horrible sexual things as a kid, and I regret everything. Can someone help me cope with myself? Can someone help me remember my past?",
      "responses": "Experiencing regret and grappling with childhood experiences can be incredibly challenging. It's important to seek professional help from a therapist or counselor who specializes in trauma or childhood issues. They can provide a safe space for you to explore your feelings, work through any trauma, and help you develop healthy coping mechanisms to navigate the challenges you're facing."
  },
  {
      "patterns": "I have been with a guy for 4 years, he's a great guy and we also have a son together. The problem is that I'm in love with a guy that I've been talking to for about 2 years but I've never met him in person. Honestly I'm bored with the relationship I have with the first guy and he makes want to go after the second guy, I don't know how to tell him that. What should I do?",
      "responses": "It's important to consider your feelings and the impact they may have on your current relationship. Communication is key in any relationship, so it would be best to have an open and honest conversation with your partner about your feelings and concerns. It's essential to address any issues or boredom you're experiencing in your current relationship before considering pursuing someone else. Relationships require effort and commitment, so it's crucial to evaluate your priorities and make a decision that aligns with your values and the well-being of everyone involved."
  },
  {
      "patterns": "My boyfriend can't get over my promiscuous past. He says he loves me but says the thought of my past is disgusting and a sexual distraction. He says he should have had more sex with different people. I don't know how to handle this.",
      "responses": "Dealing with past experiences and insecurities can be challenging in a relationship. It's important to have open and honest communication with your boyfriend about his feelings and concerns. Both of you should try to understand each other's perspectives and work towards building trust and a stronger connection. Consider seeking couples therapy or relationship counseling to facilitate these discussions and address any underlying issues. Remember, it takes time and effort from both partners to overcome obstacles and grow together."
  },
  {
      "patterns": "What do I do if I have been feeling like I could never be with anyone because no one would want me? Or I couldn't have many friends because of who I am. It's strange; I want to be loved, but I'd hate to be because I always lose.",
      "responses": "Feeling unworthy of love and experiencing self-doubt can be difficult emotions to navigate. It's essential to remember that you are deserving of love and connection, just like anyone else. Building self-confidence and a positive self-image can take time and effort. Consider seeking support from a therapist or counselor who can help you explore these feelings and develop strategies to overcome them. Additionally, surrounding yourself with supportive and understanding friends can help create a more positive social environment. Remember, you are valuable and worthy of love and friendship."
  },
  {
      "patterns": "What makes my stepchild, an 8-year-old boy, choke my daughter, a 6-year-old girl? This has been going on for 4 years. The boy tells my daughter not to tell on him. He knows it's wrong; we have tried time-outs, spanking, taking away toys, and sitting on the wall. I am afraid for my daughter's safety!",
      "responses": "I'm really sorry to hear about this difficult situation. The safety and well-being of your children should be the top priority. It's crucial to take immediate action to protect your daughter. Consider reaching out to a professional, such as a child psychologist or therapist, who can provide guidance on dealing with this specific behavior and help you establish appropriate boundaries and consequences. It's important to involve the boy's parents or legal guardians and work together to address the issue. Remember, seeking professional help can provide valuable support and guidance during challenging times."
  },
  {
      "patterns": "My coworkers are making false statements about me to the boss, saying I have mood swings. They have no proof to back it up. Can my employer make me see a therapist of their choice?",
      "responses": "Being falsely accused at work can be distressing and may impact your professional reputation. It's essential to address this situation promptly and professionally. You have the right to defend yourself and present evidence to counter the false statements made by your coworkers. Regarding the employer's request to see a therapist of their choice, it would depend on the laws and policies in your specific jurisdiction. Consider consulting with an employment lawyer or seeking advice from your local labor board to understand your rights in this situation. It may also be helpful to document any incidents or conversations related to the false accusations for future reference."
  },
  {
      "patterns": "I'm in a state of depression right now. Who can I talk to? I've been sick, in a lot of pain, and crying. Don't know where to turn.",
      "responses": "I'm really sorry to hear that you're feeling this way. It's important to reach out to someone who can provide support and understanding. Consider speaking with a mental health professional, such as a therapist or counselor, who can help you navigate through your feelings of depression and pain. Additionally, confiding in a trusted friend, family member, or a helpline specifically dedicated to mental health support can offer a listening ear and guidance. Remember, you don't have to face this alone, and seeking help is a courageous step towards healing and finding support."
  },
  {
      "patterns": "I'm a 40-year-old male and having erection problems. I still have the desire for a woman. I'm awfully frustrated. Any suggestions? My doctor doesn't want to discuss it.",
      "responses": "Experiencing difficulties with erections can be distressing and impact your overall well-being. While it's understandable that you're frustrated, it's important to address this issue with a healthcare professional who specializes in sexual health. If your current doctor doesn't want to discuss it, consider seeking a second opinion from a different healthcare provider who may be more open to discussing your concerns and exploring potential solutions. Remember, sexual health issues are common, and there are medical interventions, therapies, and lifestyle changes that can help. Don't hesitate to advocate for your own health and seek the support you need."
  },
  {
      "patterns": "My husband cheated on me, and it hurt me very bad. It was a time when my health was poor. I'm having a hard time moving on.",
      "responses": "Discovering infidelity in a relationship can be extremely painful and challenging to overcome. It's important to prioritize your emotional well-being during this time. Healing from such a betrayal takes time and self-care. Consider seeking support from a therapist or counselor who can help you process your emotions, rebuild trust, and explore options for moving forward. Surrounding yourself with a supportive network of friends and family who can provide comfort and understanding can also be beneficial. Remember, you deserve happiness and a healthy relationship, and it's possible to find healing and growth after experiencing infidelity."
  },
  {
      "patterns": "I've been emotionally abused all my life, and for some reason, I keep getting with men who emotionally abuse me. How can I stop it? I acknowledge the mistakes I've made in my life. I'm having a really hard time getting back on my feet. Can you help me, please?",
      "responses": "Experiencing emotional abuse can have long-lasting effects, and breaking free from this cycle can be challenging. It's crucial to prioritize your safety and well-being. Consider reaching out to a professional therapist or counselor who specializes in trauma and abuse. They can provide guidance, support, and strategies to help you break free from this pattern, rebuild your self-esteem, and establish healthy boundaries in future relationships. Remember, you deserve to be treated with respect, love, and kindness. Healing takes time, but with the right support and resources, it is possible to break free from the cycle of emotional abuse and build a healthier, happier life."
  },
  {
      "patterns": "I feel like I would be more comfortable as a girl, even though I still like girls. I think I'm like a girl stuck in a guy's body. I imagine myself as a girl too. I think this more because my friends say that if I were a girl, I would be a hot looking one. And I don't care about having boobs or anything. I just feel like the way that I do act will make more sense if I were a girl.",
      "responses": "Exploring your gender identity can be a complex and personal journey. It's important to remember that your feelings and experiences are valid. If you're questioning your gender identity and feeling more comfortable imagining yourself as a girl, it may be helpful to seek support from a gender therapist or counselor. They can provide guidance and a safe space for you to explore your feelings and thoughts about your gender. Remember, gender identity is unique to each individual, and there is no right or wrong way to feel. Take your time, be patient with yourself, and seek the support you need to better understand your own identity."
  },
  {
      "patterns": "I have lived with depression for some time, but things have gotten a lot better. However, why is it so hard to get over anxiety and social isolation?",
      "responses": "Recovering from depression is a significant achievement, and it's understandable to face challenges with anxiety and social isolation. Depression, anxiety, and social isolation can often be interconnected and impact each other. Overcoming anxiety and social isolation may require time, effort, and a multifaceted approach. Consider seeking support from a mental health professional who can provide strategies and techniques to manage anxiety, improve social skills, and enhance your overall well-being. Engaging in therapy, practicing self-care, and gradually exposing yourself to social situations can be helpful steps in overcoming these challenges. Remember, progress takes time, and you are not alone in this journey."
  },
  {
      "patterns": "I have a child with my baby's mother. She works, and I take care of our young son. She says she is not cheating, and I have not found anything, but she always puts me down, tells me to get out, and tells me she doesn't love me. But then the next day after our fight, she says she does. I'm having a hard time because before our child, she said she was raped by a family member, but she never went to the hospital or the cops. Now me and my family don't talk. She's always telling me I'm annoying and just belittles me. Nothing I do is right. She says, 'I work, you watch the baby.' On her days off, she never cooks or cleans. I have no friends or family, and a couple of months ago, she was confiding in some guy, but she says he's not anything to her. What do I do? I don't want to leave.",
      "responses": "It sounds like you're dealing with a challenging and complicated situation. Communication and seeking professional help are essential in addressing these issues. Consider having an open and honest conversation with your partner about your concerns, expressing how her behavior and words affect you. Relationship counseling or therapy may also be beneficial in improving communication, resolving conflicts, and addressing underlying issues. Additionally, reaching out to a support network, such as friends, family, or support groups, can provide you with emotional support and guidance. Remember, taking care of your own well-being and setting boundaries is crucial in any relationship."
  },
  {
    "patterns": "Or how to send him somewhere that can help him, something like The Baker Act.",
    "responses": "I'm sorry to hear that. If someone is in immediate danger or experiencing a mental health crisis, it's important to contact emergency services or a helpline in your country."
  },
  {
    "patterns": "In particular during family gatherings (such as funerals), where there are different customs.",
    "responses": "Family gatherings, especially during sensitive occasions like funerals, can bring about various customs and practices. It's essential to be respectful and understanding of different cultural traditions and norms."
  },
  {
    "patterns": "I don't know what's wrong with me. At times I can be really happy, excited, I'll talk fast, and I want to do things. But lately I've been staying up way later and sleeping too much. I don't hang out with friends, I don't really have them. I feel nothing, worthless. I want to do nothing, I have no interest in anything. All I usually do is stay in my room. I want to be happy, I want to be nice. What's wrong with me?",
    "responses": "It sounds like you may be experiencing a range of emotions and changes in behavior. It could be helpful to consult a mental health professional who can provide a proper evaluation and support. They can help determine any underlying issues and suggest appropriate interventions."
  },
  {
    "patterns": "I'm worried I should see a therapist due to past events and current mental status. I'm just so unsure of how to ask my doctor about seeing someone.",
    "responses": "Expressing your concerns to your doctor is an important step towards seeking help. You can try explaining your current mental state, the impact of past events, and your interest in seeing a therapist. Your doctor can guide you further and provide appropriate referrals or resources."
  },
  {
    "patterns": "I feel like my time is going too fast",
    "responses": "The feeling of time passing quickly is a common experience. It can be helpful to engage in activities that bring a sense of fulfillment and purpose, manage your time effectively, and practice mindfulness to enhance your perception of time."
  },
  {
    "patterns": "I'm worried I should see a therapist due to past events and current mental status. I'm just so unsure of how to ask my doctor about seeing someone.",
    "responses": "Expressing your concerns to your doctor is an important step towards seeking help. You can try explaining your current mental state, the impact of past events, and your interest in seeing a therapist. Your doctor can guide you further and provide appropriate referrals or resources."
  },
  {
    "patterns": "My last relationships have ended horribly. They just up and abandoned me. One of them I have never gotten closure with over it, leaving me emotionally wrecked. I know something's wrong with me but I don't know how to even start trying to better myself.I get into these depressed-like states whenever anyone I care about is too busy to be with me. I know that I shouldn't act like that because people are genuinely busy, but I feel like they're doing it just because they don't want to be around me.",
    "responses": "Experiencing difficult breakups and struggling with emotional wounds can be challenging. It may be helpful to seek support from a therapist or counselor who can provide guidance and help you navigate your emotions. They can assist you in working on self-improvement and developing healthier coping mechanisms."
  },
  {
    "patterns": "I haven't uttered one word to my boyfriend in days. Now I'm over it and don't know how to approach the situation?",
    "responses": "Communication is vital in any relationship. It's important to express your feelings and concerns to your boyfriend in a calm and open manner. Find a suitable time to have an honest conversation about your current silence, your feelings, and how you both can work towards resolving any issues."
  },
  {
    "patterns": "I've been with my boyfriend for over 3 years. He cheated on me, and the woman is now pregnant. He is physically and mentally abusive towards me. I have attempted to talk with him in order to resolve our issues; however, he never wants to talk and often blames me for everything. I'm always there when he needs my help. He tells me that although I help him, it doesn't mean that I am a good person. He always discredits me as a woman, and it's hurtful. I spend most of the day crying and feeling lonely. I am unhappy and unsure of what I need to do.",
    "responses": "I'm really sorry to hear about the difficult situation you're in. It's crucial to prioritize your safety and well-being. Consider reaching out to a trusted friend, family member, or a helpline in your country to discuss your experiences and explore potential options. They can provide guidance and support as you navigate this challenging situation."
  },
  {
    "patterns": "I started dating a guy I met online. He told me that he's a registered sex offender, but he didn't do anything. He took a plea to always register as a sex offender, and he served two years. That was more than two decades ago, and he has nothing on his record since. I always believe the past should be left in the past. But my roommates, their family, and my son are unforgiving. Should I forgive this man that I'm falling in love with, or should I dump him? If I stay with him, I will never see my family and granddaughter again.",
    "responses": "Deciding whether to forgive someone or end a relationship is a personal choice that depends on various factors. It's important to prioritize your safety, emotional well-being, and the well-being of your family. Considering the complexity of the situation, it could be beneficial to seek guidance from a professional counselor or therapist who can provide unbiased support and help you navigate your options."
  }
]

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(MyModel, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.rnn = layers.GRU(rnn_units, return_sequences=True)
        self.fc = layers.Dense(vocab_size)
    
    def call(self, inputs, training=True):
        x = self.embedding(inputs)
        x = self.rnn(x, training=training)
        x = self.fc(x)
        return x


# Extract patterns and responses from the dataset
patterns = [data["patterns"] for data in dataset]
responses = [data["responses"] for data in dataset]

# Tokenize the patterns and responses
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns + responses)

# Convert text to sequences of token IDs
pattern_sequences = tokenizer.texts_to_sequences(patterns)
response_sequences = tokenizer.texts_to_sequences(responses)

# Determine the maximum sequence length
max_sequence_length = max(max(len(seq) for seq in pattern_sequences), max(len(seq) for seq in response_sequences))

# Pad sequences to the maximum length
pattern_sequences = tf.keras.preprocessing.sequence.pad_sequences(pattern_sequences, maxlen=max_sequence_length)
response_sequences = tf.keras.preprocessing.sequence.pad_sequences(response_sequences, maxlen=max_sequence_length)

# Prepare input and target data
input_data = pattern_sequences
target_data = response_sequences

# Create TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))

# Shuffle and batch the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Define the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
rnn_units = 64

model = MyModel(vocab_size, embedding_dim, rnn_units)

# Define the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        # Make predictions
        logits = model(inputs)
        
        # Compute the loss
        loss = loss_object(targets, logits)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training loop
EPOCHS = 20
for epoch in range(EPOCHS):
    total_loss = 0.0
    
    for batch, (inputs, targets) in enumerate(dataset):
        loss = train_step(inputs, targets)
        total_loss += loss
    
    average_loss = total_loss / (batch + 1)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {average_loss:.4f}")



def generate_response_with_sampling(pattern, temperature=1.0):
    # Tokenize the pattern
    pattern_sequence = tokenizer.texts_to_sequences([pattern])
    
    # Pad the sequence to the maximum length
    pattern_sequence = tf.keras.preprocessing.sequence.pad_sequences(pattern_sequence, maxlen=max_sequence_length)
    
    # Generate response with sampling
    logits = model(pattern_sequence, training=False)
    logits = tf.squeeze(logits, axis=0)  # Remove the extra dimension
    
    probabilities = tf.nn.softmax(logits / temperature, axis=-1)
    predicted_ids = tf.random.categorical(probabilities, num_samples=1)
    
    # Convert token IDs back to text
    response = tokenizer.sequences_to_texts(predicted_ids.numpy())[0]
    
    return response



def generate_response_with_sampling(pattern, temperature=1.0):
    # Tokenize the pattern
    pattern_sequence = tokenizer.texts_to_sequences([pattern])
    
    # Pad the sequence to the maximum length
    pattern_sequence = tf.keras.preprocessing.sequence.pad_sequences(pattern_sequence, maxlen=max_sequence_length)
    
    # Generate response with sampling
    logits = model(pattern_sequence, training=False)
    logits = tf.squeeze(logits, axis=0)  # Remove the extra dimension
    
    probabilities = tf.nn.softmax(logits / temperature, axis=-1)
    predicted_ids = tf.random.categorical(tf.math.log(probabilities), num_samples=1)
    
    # Convert token IDs back to text
    response = tokenizer.sequences_to_texts(predicted_ids.numpy())[0]
    
    return response



# Example usage
pattern = "I'm feeling sad"
response = generate_response_with_sampling(pattern, temperature=0.8)
print(f"Pattern: {pattern}")
print(f"Response: {response}")

