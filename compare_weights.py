import json

main_weight_path = "epoch11_weights2.json"

input_weight_path = input("Weight Path:\n> ")

with open(main_weight_path, "r") as f:
    main_weight = json.loads(f.read())

with open(input_weight_path, "r") as f:
    input_weight = json.loads(f.read())

lora_same, lora_diff = 0, 0
base_same, base_diff = 0, 0
no_weight = 0
z = 0

# for each key in the input weight
for k, v in input_weight.items():
    if "vision_encoder" in k:
        continue
    # if it has checkpoint_wrapped_module, replace it to nothing
    key = k.replace("_checkpoint_wrapped_module.", "")
    # replace the module. prefix
    key = key.replace("module.", "")
    if key not in main_weight:
        print(key)
        no_weight += 1
        continue

    if "lora" in k:
        # print(input_weight[key], main_weight[k])
        if input_weight[k] != main_weight[key]:
            lora_diff += 1
        else:
            lora_same += 1
    else:
        if input_weight[k] != main_weight[key]:
            base_diff += 1
        else:
            base_same += 1

print(f"{lora_same=} {lora_diff=} {base_same=} {base_diff=} {no_weight=}")
