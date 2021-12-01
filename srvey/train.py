# Imports
import cfg
import data
import networks as net


def main():
    session = cfg.Session()

    if True:  # try:
        train_dataloader, val_dataloader = data.build_dataloaders()
        model = net.ArbRDNPlus(session)

        for epoch_num in range(model.curr_epoch, model.num_epochs):
            print(f"Starting epoch {epoch_num}")

            for iter_num, batch in enumerate(train_dataloader):
                # scale = batch["hr"].size / batch["lr"].size
                scale = (4, 4)
                model.set_scale(scale)
                train_dataloader.dataset.set_scale(scale)  # TODO scale option

                model.feed_data(batch)
                model.train()
                # model.train_discriminator()

                if iter_num % 5 == 0:
                    model.save_metrics()

                if iter_num % cfg.val_freq == 0:
                    for v_iter_num, batch in enumerate(val_dataloader):
                        model.feed_data(batch)
                        model.validate()
                        model.save_metrics()

                        if v_iter_num in cfg.preview_indices:
                            model.save_previews()

    # except Exception as E:
    #     print(repr(E))

    # finally:
    #     session.end()


if __name__ == "__main__":
    main()
