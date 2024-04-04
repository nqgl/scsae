from nqgl.hsae_re.data.stored_acts_buffer import ActsConfig, store_acts

ac = ActsConfig(2, 5, dtype="fp16", exclude_first_acts=False)


def main():
    store_acts(ac)


if __name__ == "__main__":
    main()
