import sys
import jsonlines


def main():
    input_path, output_path = sys.argv[1], sys.argv[2]

    input_f = open(input_path)
    output_f = jsonlines.open(output_path, 'w')

    for text in input_f.read().splitlines():
        output_f.write({'text': text})

    return


if __name__ == '__main__':
    main()
