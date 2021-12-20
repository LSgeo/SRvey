import cfg
import data
import networks as net


def main():
    session = cfg.Session()


    if True:  # try:
        train_dataloader, val_dataloader, preview_dataloader = data.build_dataloaders(
            iters_per_epoch=cfg.iters_per_epoch
        )
        model = net.ArbRDNPlus(session)
        # model = net.RDNPlus(session)

        for epoch_num in range(model.start_epoch, model.num_epochs):
            session.begin_epoch(epoch_num)
            model.curr_epoch += 1

            for iter_num, batch in enumerate(train_dataloader):
                if iter_num == cfg.iters_per_epoch:
                    break
                model.curr_iteration += 1
                # scale = batch["hr"].size / batch["lr"].size
                scale = (4, 4)
                model.set_scale(scale)
                train_dataloader.dataset.set_scale(scale)  # TODO scale option

                model.feed_data(batch)
                model.train()

                if iter_num % 100 == 0 and iter_num != 0:
                    model.log_metrics(log_to_comet=True)

                if iter_num % cfg.val_freq == 0 and iter_num != 0:
                    for i, batch in enumerate(val_dataloader):
                        if i == cfg.iters_per_epoch // 4:
                            break
                        model.feed_data(batch)
                        model.validate()
                        model.log_metrics(log_to_comet=True)

                if iter_num % cfg.preview_freq == 0:
                    for batch in preview_dataloader:
                        model.feed_data(batch)
                        model.save_previews(log_to_comet=True)

                

    # except Exception as E:
    #     print(repr(E))

    # finally:
    #     session.end()


if __name__ == "__main__":
    main()
