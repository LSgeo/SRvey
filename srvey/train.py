def main():
    import cfg
    import data
    import networks as net

    session = cfg.Session(debug=True)

    # try:
    if 1:
        train_dataloader, val_dataloader, preview_dataloader = data.build_dataloaders()
        model = net.ArbRDNPlus(session, len(train_dataloader) * cfg.num_epochs)
        # model = net.RDNPlus(session, len(train_dataloader) * cfg.num_epochs)  # Locked 4x sr

        if cfg.pretrained_model:
            model.load_pretrained_model(cfg.pretrained_model)
            # Validate first if loading pre-trained model TODO

        for epoch_num in range(model.start_epoch, model.num_epochs):
            session.begin_epoch(epoch_num)
            model.curr_epoch += 1

            for iter_num, batch in enumerate(train_dataloader):
                model.curr_iteration += 1

                scale = (4, 4)
                model.set_scale(scale)
                train_dataloader.dataset.set_scale(scale)  # TODO scale option

                model.feed_data(batch)
                model.train()

                if iter_num % cfg.metric_freq == 0 and iter_num != 0:
                    model.log_metrics(log_to_comet=True)

            if epoch_num % cfg.val_freq == 0:
                for batch in val_dataloader:
                    model.feed_data(batch)
                    model.validate()
                    model.log_metrics(log_to_comet=True)

            if epoch_num % cfg.preview_freq == 0:
                for batch in preview_dataloader:
                    model.feed_data(batch)
                    model.save_previews(log_to_comet=True)

            if epoch_num % cfg.checkpoint_freq == 0:
                model.save_model(for_inference_only=False)

    # except Exception as E:
    #     print(repr(E))

    # finally:
    if 1:
        # model.save_model(name=f"final_epoch_{epoch_num}.tar", for_inference_only=False)
        session.end()


if __name__ == "__main__":
    main()
