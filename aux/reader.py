import os
import pandas as pd

emotion_map = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry',
               6: 'fearful', 7 : 'disgust', 8 : 'surprised'}

source_map = {1 : 'speech', 2 : 'song'}


def read_data(data_path):
    # read data from data path

    dir_list = os.listdir(data_path)
    dir_list.sort()

    if '.DS_Store' in dir_list:
        dir_list.remove('.DS_Store')

    data_df = pd.DataFrame(columns=['path', 'source', 'actor', 'gender',
                                    'intensity', 'statement', 'repetition',
                                    'emotion'])
    count = 0
    for i in dir_list:
        file_list = os.listdir(data_path + i)
        for f in file_list:
            nm = f.split('.')[0].split('-')

            if len(nm) == 7:
                path = i + '/' + f
                src = int(nm[1])
                actor = int(nm[-1])
                emotion = int(nm[2])

                if int(actor)%2 == 0:
                    gender = "female"
                else:
                    gender = "male"

                if nm[3] == '01':
                    intensity = 0
                else:
                    intensity = 1

                if nm[4] == '01':
                    statement = 0
                else:
                    statement = 1

                if nm[5] == '01':
                    repeat = 0
                else:
                    repeat = 1

                data_df.loc[count] = [path, src, actor, gender, intensity,
                                      statement, repeat, emotion]
                count += 1

    return data_df.replace({"emotion": emotion_map, "source": source_map})
