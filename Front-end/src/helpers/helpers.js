import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import isSameOrAfter from 'dayjs/plugin/isSameOrAfter';

dayjs.extend(relativeTime);
dayjs.extend(isSameOrAfter);

export const formatTime = (dateString) => {
  const now = dayjs();
  const date = dayjs(dateString);

  const diffDays = now.diff(date, 'day');

  if (diffDays < 7) {
    return date.fromNow(); // e.g. "3 days ago"
  } else {
    return date.format('YYYY-MM-DD HH:mm');
  }
};

export const formatExactTime = (dateString) => {
  const date = dayjs(dateString);

  return date.format('YYYY-MM-DD HH:mm');
};
