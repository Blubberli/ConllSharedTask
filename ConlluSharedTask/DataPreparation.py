#!/usr/bin/env python3.6
import re
import pickle


def extract_pairs(ud_file, output_file, output_file2):

    nn1_input = []
    nn2_input = []

    root = ["ROOT", "ROOT", "ROOT", -1, "ROOT"]
    start1 = ["START-1", "START-1", "START-1", -1, "START-1"]
    start2 = ["START-2", "START-2", "START-2", -1, "START-2"]
    end1 = ["END+1", "END+1", "END+1", -1, "END+1"]
    end2 = ["END+2", "END+2", "END+2", -1, "END+2"]

    sent = [root]
    tree = []

    with open(ud_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if line:
                if not line.startswith("#"):

                    split = re.split(r'\s+', line)

                    if len(split) >= 7:
                        if "-" not in split[0]:
                            word = [split[1], split[3], split[5], int(split[6]), split[7]]
                            sent.append(word)
            else:
                context_left12 = None
                context_left11 = None

                context_right11 = None
                context_right12 = None

                for dependent in range(1, len(sent)):

                    word1 = sent[dependent]

                    if len(sent) == 2:
                        context_left12 = start2
                        context_left11 = start1

                        context_right11 = end1
                        context_right12 = end2

                    elif len(sent) == 3:

                        if dependent == 1:
                            context_left12 = start2
                            context_left11 = start1

                            context_right11 = sent[2]
                            context_right12 = end1

                        elif dependent == 2:
                            context_left12 = start1
                            context_left11 = sent[1]

                            context_right11 = end1
                            context_right12 = end2

                    elif len(sent) > 3:

                        if dependent == 1:
                            context_left12 = start2
                            context_left11 = start1
                            context_right11 = sent[2]
                            context_right12 = sent[3]

                        elif dependent == len(sent) - 2:
                            context_left12 = context_left11
                            context_left11 = sent[dependent - 1]

                            context_right11 = sent[dependent + 1]
                            context_right12 = end1

                        elif dependent == len(sent) - 1:
                            context_left12 = context_left11
                            context_left11 = sent[dependent - 1]

                            context_right11 = end1
                            context_right12 = end2

                        elif 0 < dependent < len(sent) - 2:
                            context_left12 = context_left11
                            context_left11 = sent[dependent - 1]

                            context_right11 = sent[dependent + 1]
                            context_right12 = sent[dependent + 2]

                    context_left22 = None
                    context_left21 = None

                    context_right21 = None
                    context_right22 = None

                    for head in range(len(sent)):

                        word2 = sent[head]

                        if head == 0:
                            context_left21 = root
                            context_left22 = root

                            context_right21 = root
                            context_right22 = root

                        else:
                            if len(sent) == 2:
                                context_left22 = start2
                                context_left21 = start1

                                context_right21 = end1
                                context_right22 = end2

                            elif len(sent) == 3:

                                if head == 1:
                                    context_left22 = start2
                                    context_left21 = start1

                                    context_right21 = sent[2]
                                    context_right22 = end1

                                elif head == 2:
                                    context_left22 = start1
                                    context_left21 = sent[0]

                                    context_right21 = end1
                                    context_right22 = end2

                            elif len(sent) > 3:

                                if head == 1:
                                    context_left22 = start2
                                    context_left21 = start1
                                    context_right21 = sent[2]
                                    context_right22 = sent[3]

                                elif head == len(sent) - 2:
                                    context_left22 = context_left21
                                    context_left21 = sent[head - 1]

                                    context_right21 = sent[head + 1]
                                    context_right22 = end1

                                elif head == len(sent) - 1:
                                    context_left22 = context_left21
                                    context_left21 = sent[head - 1]

                                    context_right21 = end1
                                    context_right22 = end2

                                elif 1 < head < len(sent) - 2:
                                    context_left22 = context_left21
                                    context_left21 = sent[head - 1]

                                    context_right21 = sent[head + 1]
                                    context_right22 = sent[head + 2]

                        word1_context = [context_left12[0], context_left11[0],
                                         word1[0],
                                         context_right11[0], context_right12[0]]
                        pos1_context = [context_left12[1], context_left11[1],
                                        word1[1],
                                        context_right11[1], context_right12[1]]
                        morph1_context = [context_left12[2], context_left11[2],
                                          word1[2],
                                          context_right11[2], context_right12[2]]

                        word2_context = [context_left22[0], context_left21[0],
                                         word2[0],
                                         context_right21[0], context_right22[0]]
                        pos2_context = [context_left22[1], context_left21[1],
                                        word2[1],
                                        context_right21[1], context_right22[1]]
                        morph2_context = [context_left22[2], context_left21[2],
                                          word2[2],
                                          context_right21[2], context_right22[2]]

                        is_connected = (word1[3] == head)

                        distance = head - dependent

                        if head == 0:
                            distance = 0

                        label = "NONE"

                        if is_connected:
                            label = word1[4]

                        combined = [word1_context, pos1_context, morph1_context,
                                    word2_context, pos2_context, morph2_context,
                                    is_connected,
                                    label,
                                    distance]
                        nn1_input.append(combined)

                    cur = word1
                    nxt = cur[3]
                    path_to_word = []

                    while nxt != -1:
                        tmp = [cur[0], sent[nxt][0], cur[4]]
                        path_to_word.append(tmp)

                        cur = sent[nxt]
                        nxt = cur[3]

                    tree.append(path_to_word)

                removable = []

                truncated_tree = []

                for partial_path in tree:
                    for connection in partial_path:
                        if connection not in removable:

                            truncated_tree.append(connection)
                            removable.append(connection)

                print(truncated_tree)

                nn2_input.append(truncated_tree)

                tree = []
                sent = [root]

    pickle.dump(nn1_input, open(output_file, 'wb'))
    pickle.dump(nn2_input, open(output_file2, 'wb'))
