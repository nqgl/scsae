import sys

# sys.path.append("./nqgl/hsae_re/")
from nqgl.hsae_re.data.stored_acts_buffer import ActsConfig, store_acts

ac = ActsConfig(
    2,
    5,
    dtype="fp32",
    storage_dtype="fp16",
    exclude_first_acts=False,
    set_bos=True,
    model_name="gpt2",
)


def main():
    store_acts(ac)


if __name__ == "__main__":
    main()
