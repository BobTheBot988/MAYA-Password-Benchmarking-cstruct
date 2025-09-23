from script.utils.file_operations import save_pickle

def count_to_plain(input_file, output_file):
    passwords = []
    non_utf8_passwords = 0

    with open(input_file, 'rb') as fi:
        for line in fi:
            error_flag = False
            try:
                line = line.decode(encoding='utf-8')
            except UnicodeDecodeError:
                line = line.decode(encoding='utf-8', errors='ignore')
                error_flag = True

            try:
                line = line.strip()
                count, password = line.split(' ', 1)
                if error_flag:
                    non_utf8_passwords += int(count)
                password = password.strip("\n")
                password = password.replace(" ", "")

                if password:
                    for i in range(int(count)):
                        passwords.append(password + "\n")

            except ValueError:
                continue

    save_pickle(output_file, passwords)

    print(f"Total passwords: {len(passwords)}.")
    print(f"# non-UTF8 passwords: {non_utf8_passwords}.")
    print(f"% of non-UTF8 passwords: {non_utf8_passwords / len(passwords) * 100}.")


def email_to_plain(input_file, output_file, mode):
    passwords = []
    non_utf8_passwords = 0

    with open(input_file, 'rb') as fi:
        for line in fi:
            try:
                line = line.decode(encoding='utf-8')
            except UnicodeDecodeError:
                line = line.decode(encoding='utf-8', errors='ignore')
                non_utf8_passwords += 1

            try:
                if mode == "first":
                    index = line.find(":")
                elif mode == "last":
                    index = line.rfind(":")
                elif mode == "second":
                    first_index = line.find(":")
                    index = line.find(":", first_index + 1)

                if (index == -1):
                    continue

                password = line[index + 1:-1]
                password = password.rstrip("\n")
                password = password.replace(" ", "")

                if password:
                    passwords.append(password + "\n")

            except ValueError:
                continue

    # Save the passwords list to a pickle file
    save_pickle(output_file, passwords)

    print(f"Total passwords: {len(passwords)}.")
    print(f"# non-UTF8 passwords: {non_utf8_passwords}")
    print(f"% of non-UTF8 passwords: {non_utf8_passwords / len(passwords) * 100}.")


def format_plain(input_file, output_file):
    passwords = []
    non_utf8_passwords = 0

    with open(input_file, 'rb') as fi:
        for line in fi:
            try:
                password = line.decode(encoding='utf-8')
            except UnicodeDecodeError:
                password = line.decode(encoding='utf-8', errors='ignore')
                non_utf8_passwords += 1

            try:
                password = password.replace(" ", "")
                password = password.strip("\n")
                if password:
                    passwords.append(password + "\n")

            except ValueError:
                continue

    save_pickle(output_file, passwords)

    print(f"Total passwords: {len(passwords)}.")
    print(f"# non-UTF8 passwords: {non_utf8_passwords}")
    print(f"% of non-UTF8 passwords: {non_utf8_passwords / len(passwords) * 100}.")
