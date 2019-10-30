#!/usr/bin/env bash
NAME=$1
SEQ=$2

if [[ "$2" == "seq" ]]
    then
        echo "Seq option is chosen!"
        for name in 1 37 72 70 44
        do
            echo $name
            python pick_image.py --name $name --seq True
            python create_json_file.py --name $name
            python build_vocab.py --name $name
        done
        id

        python pick_image.py --name $NAME
        python create_json_file.py --name $NAME
        python build_vocab.py --name $NAME
        id
elif [[ "$2" == "1" ]]
    then
        echo "One option is chosen!"
        for name in 1
        do
            echo $name
            python pick_image.py --name $name --seq False
            python create_json_file.py --name $name
            python build_vocab.py --name $name
         done
        id

elif [[ "$2" == "once" ]]
    then
        echo "Once option is chosen!"
        for name in once
        do
            echo $name
            python pick_image.py --name $name --seq False
            python create_json_file.py --name $name
            python build_vocab.py --name $name
        done
        id

elif [[ "$2" == "2to21" ]]
    then
        echo "2to21 option is chosen!"
        for name in 2to21
        do
            echo $name
            python pick_image.py --name $name --seq False
            python create_json_file.py --name $name
            python build_vocab.py --name $name
        done
        id
fi
