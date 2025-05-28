from flask import Blueprint

# 创建Blueprint
news_bp = Blueprint('news', __name__)
# 创建蓝图
images_bp = Blueprint('images', __name__)
novo_bp = Blueprint('novo', __name__)
pixtock_bp = Blueprint('pixtock', __name__)
set_weights_bp = Blueprint('set_weights', __name__)
user_bp = Blueprint('user', __name__)

from . import images, novo, pixtock, user, SetWeights, news