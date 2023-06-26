import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self, filename, printflag=False, level='info', when='D', backCount=3, fmt='%(asctime)s - %(message)s'):
        self.logger = logging.getLogger(filename)
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.logger.setLevel(self.level_relations.get(level))
        if printflag:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(formatter)
        self.logger.addHandler(th)


