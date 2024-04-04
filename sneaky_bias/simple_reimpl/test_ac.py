from nqgl.hsae_re.sneaky_bias.simple_reimpl import test, ac_cfg

ac = ac_cfg.ac
trainer = test.trainer

trainer.train(ac.read_as_iter(2048))
