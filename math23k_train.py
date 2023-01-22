# coding: utf-8
from src.train_and_evaluate_tail import *
from src.models_tail import *
import time
import torch.optim
from src.expressions_transfer import *
import random
import sys

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

torch.cuda.set_device(1)
batch_size = 32
embedding_size = 128
hidden_size = 512
n_epochs = 180
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'
if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    seed = 2
torch.manual_seed(seed)
random.seed(seed)

print('seed:', seed)
def get_train_test_fold(ori_path,prefix,data,pairs,group):
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]
    train_fold = []
    valid_fold = []
    test_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)
    return train_fold, test_fold, valid_fold

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num


data = load_raw_data("data/Math_23K.json")
group_data = read_json("data/Math_23K_processed.json")


pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []
for p in pairs:
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3], p[4]))
pairs = temp_pairs


#train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)


train_fold, valid_fold, test_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)

best_acc_fold = []

pairs_tested = test_fold
#pairs_trained = valid_fold
pairs_trained = train_fold

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)

d = {}
dd = {}
ddd = {}
for idx, iii in enumerate(train_pairs):
    op_list = [0,1,2,3]
    output_string = ''.join([str(t) for t in iii[2]])
    if iii[2][0] in op_list:
        if output_string[0] not in d:
            d[output_string[0]] = [idx]
        else:
            d[output_string[0]].append(idx)
    if iii[2][0] in op_list and iii[2][1] in op_list:
        if output_string[:2] not in dd:
            dd[output_string[:2]] = [idx]
        else:
            dd[output_string[:2]].append(idx)
    if iii[2][0] in op_list and iii[2][1] in op_list and iii[2][2] in op_list:
        if output_string[:3] not in ddd:
            ddd[output_string[:3]] = [idx]
        else:
            ddd[output_string[:3]].append(idx)




# Initialize models
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                        n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
first_dis = Discriminator(hidden_size=hidden_size)
second_dis = Discriminator(hidden_size=hidden_size)
seq_encoder = EncoderRNN(input_size = output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size, n_layers=n_layers, dropout=0.05)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
first_dis_optimizer = torch.optim.Adam(first_dis.parameters(), lr=learning_rate, weight_decay=weight_decay)
second_dis_optimizer = torch.optim.Adam(second_dis.parameters(), lr=learning_rate, weight_decay=weight_decay)
seq_encoder_optimizer = torch.optim.Adam(seq_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=30, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=30, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=30, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=30, gamma=0.5)
first_scheduler = torch.optim.lr_scheduler.StepLR(first_dis_optimizer, step_size=30, gamma=0.5)
second_scheduler = torch.optim.lr_scheduler.StepLR(second_dis_optimizer, step_size=30, gamma=0.5)
seq_encoder_scheduler = torch.optim.lr_scheduler.StepLR(seq_encoder_optimizer, step_size=30, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    first_dis.cuda()
    second_dis.cuda()
    seq_encoder.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])


for epoch in range(n_epochs):

    loss_total = 0
    loss_dis = 0
    loss_dis_total = 0
    loss_cl_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):
        first_random = random.choice(list(d.keys()))
        first_pos = random.sample(d[first_random], min(int(batch_size),len(d[first_random])))
        temp = []
        for sub_list in d:
            if sub_list != first_random:
                temp = temp + d[sub_list]
        first_neg = random.sample(temp, min(int(batch_size),len(d[first_random])))

        second_random = random.choice(list(dd.keys()))
        second_pos = random.sample(dd[second_random], min(int(batch_size),len(dd[second_random])))
        temp = []
        for sub_list in dd:
            if sub_list != second_random:
                temp = temp + dd[sub_list]
        second_neg = random.sample(temp, min(int(batch_size),len(dd[second_random])))
        first_pos = copy_list([train_pairs[i][:2] for i in first_pos])
        first_neg = copy_list([train_pairs[i][:2] for i in first_neg])
        second_pos = copy_list([train_pairs[i][:2] for i in second_pos])
        second_neg = copy_list([train_pairs[i][:2] for i in second_neg])
        #if True:
        #    for _ in range(5):
    
        loss, loss_dis, loss_cl = train_tree_all_together(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge, first_dis, second_dis, seq_encoder,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, first_dis_optimizer, second_dis_optimizer, seq_encoder_optimizer, first_pos, first_neg, second_pos, second_neg, 
            output_lang, num_pos_batches[idx], epoch)
        loss_total += loss
        loss_dis_total += loss_dis
        loss_cl_total += loss_cl
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    first_scheduler.step()
    second_scheduler.step()
    seq_encoder_scheduler.step()
    print("loss:", loss_total / len(input_lengths))
    print('loss_dis:', loss_dis_total / len(input_lengths))
    print('loss_cl:', loss_cl_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    if epoch % 5 == 0 or epoch > n_epochs - 10:
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                        merge, output_lang, test_batch[5], beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        if epoch > n_epochs - 10 or epoch == 0:
            torch.save(encoder.state_dict(), "models/encoder_all_" + str(epoch) + '_seed_' + str(seed))
            torch.save(predict.state_dict(), "models/predict_all_"+ str(epoch) + '_seed_' + str(seed))
            torch.save(generate.state_dict(), "models/generate_all_"+ str(epoch) + '_seed_' + str(seed))
            torch.save(merge.state_dict(), "models/merge_all_"+ str(epoch) + '_seed_' + str(seed))


