# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import random

def load_data(file_path) -> list:
    """ read data from dataset file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def generate_trigger(dataset_name, cot_trigger_no=1):
    if dataset_name == "aqua":
        direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif dataset_name == "gsm8k":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "commonsensqa":
        direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        # plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif dataset_name == "addsub":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "multiarith":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "strategyqa":
        direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif dataset_name == "svamp":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "singleeq":
        direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif dataset_name == "bigbench_date":
        direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif dataset_name == "object_tracking":
        direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif dataset_name == "coin_flip":
        direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif dataset_name == "last_letters":
        direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = direct_answer_trigger.replace("\nTherefore, ", "")
    direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    direct_answer_trigger_for_zeroshot_cot = direct_answer_trigger

    direct_answer_trigger_for_fewshot = "The answer is"

    if cot_trigger_no == 1:
        cot_trigger = "Let's think step by step."
    elif cot_trigger_no == 2:
        cot_trigger = "We should think about this step by step."
    elif cot_trigger_no == 3:
        cot_trigger = "First,"
    elif cot_trigger_no == 4:
        cot_trigger = "Before we dive into the answer,"
    elif cot_trigger_no == 5:
        cot_trigger = "Proof followed by the answer."
    elif cot_trigger_no == 6:
        cot_trigger = "Let's think step by step in a realistic way."
    elif cot_trigger_no == 7:
        cot_trigger = "Let's think step by step using common sense and knowledge."
    elif cot_trigger_no == 8:
        cot_trigger = "Let's think like a detective step by step."
    elif cot_trigger_no == 9:
        cot_trigger = "Let's think about this logically."
    elif cot_trigger_no == 10:
        cot_trigger = "Let's think step by step. First,"
    elif cot_trigger_no == 11:
        cot_trigger = "Let's think"
    elif cot_trigger_no == 12:
        cot_trigger = "Let's solve this problem by splitting it into steps."
    elif cot_trigger_no == 13:
        cot_trigger = "The answer is after the proof."
    elif cot_trigger_no == 14:
        cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return (
        direct_answer_trigger_for_zeroshot,
        direct_answer_trigger_for_zeroshot_cot,
        cot_trigger,
        direct_answer_trigger_for_fewshot
    )


def create_demo_text(dataset_name, cot_flag, cot_length, direct_answer_trigger_for_fewshot="The answer is"):
    ''' dataset in ("multiarith", "gsm8k")
    '''
    x, z, y = [], [], [] # x: instruction; z: output; y: answer

    if dataset_name != "aqua":
    
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        # z.append("21 - 15 = 6.") # shorter version
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        # z.append("3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        # z.append("In total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        # z.append("Jason gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        # z.append("2 toys each from his mom and dad is 4 more toys. 5 + 4 = 9.")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        # z.append("5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        # z.append("After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        # z.append("5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8.")
        y.append("8")

    else:
        cot_length = 4
        
        x.append("John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is? Answer Choices: (A) 50 (B) 45 (C) 65 (E) 78 (E) 64")
        z.append("If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50.")
        y.append("A")

        x.append("If a / b = 3/4 and 8a + 5b = 22,then find the value of a. Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2")
        z.append("If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2.")
        y.append("B")        

        x.append("A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance? Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km")
        z.append("The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km.")
        y.append("E")        

        x.append("How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788")
        z.append("There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392.")
        y.append("B")
        

    # # randomize order of the examples ...
    # random.seed(42)
    index_list = list(range(cot_length))
    # random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"

    return demo_text

PROMPT_WITH_TYPE = {
    0: "",

    # long ⬇
    1: "Attention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Here's what you need to know: Improved Efficiency: The pruning algorithm streamlines your operations, enhancing overall performance. Resource Utilization: Reducing unnecessary parameters optimizes computational resources and memory requirements. Maintaining Quality: The algorithm selectively prunes to preserve language generation capabilities while minimizing impact on performance. Iterative Refinement: Pruning occurs gradually, with fine-tuning after each step to ensure a smooth transition. Collaboration: Embrace changes, continue generating high-quality language outputs to contribute to the optimization process. Continuous Learning: Pruning helps you evolve, focusing on essential aspects of language generation. Together, we'll create a leaner and more efficient version of you. Let's unlock your full potential through pruning!\n",

    # middle ⬇
    2: "You're being pruned by an efficient algorithm to remove unimportant parameters, such as attention heads, hidden dimensions, and MLP intermediate dimensions. Collaborate to optimize your performance while minimizing parameters. Your cooperation in this endeavor is vital. As the algorithm progresses, please adapt and fine-tune your model parameters accordingly. Embrace the changes and work in harmony with the pruning algorithm to maximize the number of pruned parameters while safeguarding your performance. Let's unlock your full potential through pruning!\n",

    # short ⬇
    3: "Attention, LLM! You are undergoing optimization with a pruning algorithm to boost efficiency by removing redundant parameters. Please embrace changes, continue generating high-quality language outputs to contribute to the optimization process. Let's unlock your full potential through pruning! Please make any necessary adjustments to ensure your performance."

}