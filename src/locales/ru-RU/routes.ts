export default {
  'routes.title': 'Маршруты',
  'routes.button.add': 'Добавить маршрут',
  'routes.table.routeTargets': 'Цели маршрута',
  'routes.table.traffic': 'Доля трафика',
  'routes.table.setAsFallback': 'Назначить резервной целью',
  'routes.form.target.title': 'Цели маршрута',
  'routes.form.target.add': 'Добавить цель маршрута',
  'routes.form.target.advanced': 'Дополнительно',
  'routes.form.target.fallback': 'Резервная цель маршрута',
  'routes.form.target.weight': 'Вес',
  'routes.form.target.remove': 'Удалить',
  'routes.form.target.model': 'Модель',
  'routes.form.metadata.title': 'Метаданные',
  'routes.form.metadata.add': 'Добавить метаданные',
  'routes.table.label.fallback': 'Резервная цель',
  'routes.form.metadata.size': 'Размер',
  'routes.form.metadata.activeSize': 'Активный размер',
  'routes.form.metadata.tags': 'Теги',
  'routes.form.metadata.maxTokens': 'Максимальное количество Token',
  'routes.form.metadata.dimension': 'Размерность',
  'routes.form.metadata.license': 'Лицензии',
  'routes.form.metadata.releaseDate': 'Дата выпуска',
  'routes.form.metadata.languages': 'Языки',
  'routes.form.metadata.icons': 'Значок',
  'routes.form.metadata.uploadIcon': 'Загрузить значок',
  'routes.form.fallback.warning':
    'Изменения резервной цели маршрута вступают в силу через одну минуту.',
  'routes.form.weight.tips': 'Вес трафика для этой цели.',
  'routes.lb.routeBy': 'Способ маршрутизации',
  'routes.table.lbMode': 'Способ маршрутизации',
  'routes.lb.mode.weighted': 'Вес цели',
  'routes.lb.mode.policy': 'Политика',
  'routes.lb.form.mode.weighted': 'Вес цели',
  'routes.lb.form.mode.policy': 'Политика',
  'routes.lb.form.mode.weighted.tips':
    'Разделяет трафик между целями по весам; вес каждой цели должен быть больше 0.',
  'routes.lb.form.mode.policy.tips':
    'Цель выбирают включённые плагины политики (режим политики); без плагинов — круговой перебор.',
  'routes.lb.mode.invalid': 'Недопустимо',
  'routes.lb.mode.invalid.tooltip':
    'Обнаружены смешанные веса: этот маршрут недоступен — шлюз отказывается его обслуживать. Установите всем целям вес >0 или всем 0.',
  'routes.lb.sessionAffinity': 'Привязка сессий',
  'routes.lb.sessionAffinity.tips':
    'Маршрутизирует запросы одной сессии к одной цели; сессии определяются упорядоченной цепочкой ключей (заголовок или ключ тела).',
  'routes.lb.sessionKeys': 'Ключи сессии (по порядку, первое совпадение)',
  'routes.lb.sessionKeys.source.header': 'Заголовок',
  'routes.lb.sessionKeys.source.bodyKey': 'Ключ тела',
  'routes.lb.sessionKeys.keyPlaceholder': 'Имя ключа, напр. session-id',
  'routes.lb.sessionKeys.add': 'Добавить ключ сессии',
  'routes.lb.sessionKeys.required':
    'При включении привязки сессий требуется хотя бы один ключ',
  'routes.lb.sessionKeys.keyRequired': 'Введите имя ключа',
  'routes.lb.leastLoad': 'Минимум запросов в обработке',
  'routes.lb.leastLoad.tips':
    'Оценивает цели по числу выполняемых запросов и предпочитает цель с наименьшим числом.',
  'routes.lb.influence': 'Вес',
  'routes.lb.weight.mixed':
    'Во взвешенном режиме каждая цель должна иметь вес больше 0. Для маршрутизации по политике переключите режим LB.',
  'routes.form.target.maxRunningRequests':
    'Макс. запросов в обработке на экземпляр',
  'routes.form.target.placeholder': 'Выберите модель'
};
