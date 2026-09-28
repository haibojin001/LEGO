from argostranslate import package, translate
from packaging import version
from minisbd import download_models
import libretranslate.language


def boot(load_only=None, update_models=False, install_models=False):
    try:
        if update_models:
            check_and_install_models(
                load_only_lang_codes=load_only,
                update=update_models,
            )
        else:
            check_and_install_models(
                force=install_models,
                load_only_lang_codes=load_only,
            )
    except Exception as error:
        print("Cannot update models (normal if you're offline): %s" % str(error))


def check_and_install_models(force=False, load_only_lang_codes=None, update=False):
    should_install = len(package.get_installed_packages()) < 2 or force or update

    if not should_install:
        return

    print("Updating language models")
    package.update_package_index()

    available_packages = package.get_available_packages()
    installed_packages = package.get_installed_packages()

    print("Found %s models" % len(available_packages))

    if load_only_lang_codes is not None:
        load_only_lang_codes = libretranslate.language.iso2model(
            load_only_lang_codes
        )

        missing_codes = set(load_only_lang_codes)
        for model_package in available_packages:
            missing_codes.difference_update(
                {model_package.from_code, model_package.to_code}
            )

        if missing_codes:
            raise ValueError(
                "Unavailable language codes: %s."
                % ",".join(sorted(missing_codes))
            )

        available_packages = [
            model_package
            for model_package in available_packages
            if model_package.from_code in load_only_lang_codes
            and model_package.to_code in load_only_lang_codes
        ]

        if not available_packages:
            raise ValueError("no available package")

        print("Keep %s models" % len(available_packages))

    for available_package in available_packages:
        already_present = False

        if not force:
            for installed_package in installed_packages:
                same_direction = (
                    installed_package.from_code == available_package.from_code
                    and installed_package.to_code == available_package.to_code
                )

                if same_direction:
                    already_present = True

                    installed_version = version.parse(
                        installed_package.package_version
                    )
                    available_version = version.parse(
                        available_package.package_version
                    )

                    if installed_version < available_version:
                        print(
                            f"Updating {available_package} "
                            f"({installed_package.package_version}->"
                            f"{available_package.package_version}) ..."
                        )
                        installed_package.update()

        if not already_present:
            print(
                f"Downloading {available_package} "
                f"({available_package.package_version}) ..."
            )
            available_package.install()

    print("Downloading MiniSBD models")
    download_models(load_only_lang_codes, print)

    libretranslate.language.languages = translate.get_installed_languages()
    print(
        f"Loaded support for {len(translate.get_installed_languages())} languages "
        f"({len(available_packages)} models total)!"
    )