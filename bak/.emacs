(setq backup-directory-alist `((".*" . ,"~/.saves")))
(setq create-lockfiles nil)

(custom-set-variables
 '(column-number-mode t)
 '(tool-bar-mode nil)
 '(menu-bar-mode nil))

(global-font-lock-mode t)
(setq-default indent-tabs-mode nil)
(setq indent-tabs-mode nil)

(setq require-final-newline t)

(setq c-basic-offset 2)
(setq sgml-basic-offset 4)
(setq next-line-add-newlines nil)

(global-set-key "\C-cg" 'goto-line)
(global-set-key "\C-m" 'newline-and-indent)

(setq default-tab-width 4)
(setq inhibit-splash-screen t)
(add-to-list 'auto-mode-alist '("\\.cu\\'" . c-mode))
