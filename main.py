from SentenceReadingAgent import SentenceReadingAgent

def test():
    #This will test your SentenceReadingAgent
	#with nine initial test cases.

    test_agent = SentenceReadingAgent()

    sentence_1 = "Ada brought a short note to Irene."
    question_1 = "Who brought the note?"
    question_2 = "What did Ada bring?"
    question_3 = "Who did Ada bring the note to?"
    question_4 = "How long was the note?"

    sentence_2 = "David and Lucy walk one mile to go to school every day at 8:00AM when there is no snow."
    question_5 = "Who does Lucy go to school with?"
    question_6 = "Where do David and Lucy go?"
    question_7 = "How far do David and Lucy walk?"
    question_8 = "How do David and Lucy get to school?"
    question_9 = "At what time do David and Lucy walk to school?"

    sentence_3 = "The white dog and the blue horse play together."
    question_10 = "What animal is blue?"

    sentence_4 = "She will write him a love letter."
    question_11 = "Who wrote a love letter?"

    sentence_5 = "Serena and Ada took the blue rock to the street."
    question_12 = "Where did they take the rock?"


    print(test_agent.solve(sentence_1, question_1))  # "Ada"
    print(test_agent.solve(sentence_1, question_2))  # "note" or "a note"
    print(test_agent.solve(sentence_1, question_3))  # "Irene"
    print(test_agent.solve(sentence_1, question_4))  # "short"

    print(test_agent.solve(sentence_2, question_5))  # "David"
    print(test_agent.solve(sentence_2, question_6))  # "school"
    print(test_agent.solve(sentence_2, question_7))  # "mile" or "a mile"
    print(test_agent.solve(sentence_2, question_8))  # "walk"
    print(test_agent.solve(sentence_2, question_9))  # "8:00AM"

    print(test_agent.solve(sentence_3, question_10))  # "horse"

    print(test_agent.solve(sentence_4, question_11))  # "She"

    print(test_agent.solve(sentence_5, question_12))  # "street"

if __name__ == "__main__":
    test()